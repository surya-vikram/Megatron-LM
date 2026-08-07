# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import glob
import multiprocessing
import queue
import traceback
from pathlib import Path

import numpy as np
try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class
    nltk_available = False

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


SUPPORTED_INPUT_SUFFIXES = {'.jsonl', '.parquet'}


def discover_input_files(input_path):
    """Resolve a file, glob, or directory into deterministic input paths."""
    path = Path(input_path)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = [candidate for candidate in path.rglob('*') if candidate.is_file()]
    else:
        candidates = []
        for match in glob.glob(input_path, recursive=True):
            match_path = Path(match)
            if match_path.is_dir():
                candidates.extend(candidate for candidate in match_path.rglob('*') if candidate.is_file())
            elif match_path.is_file():
                candidates.append(match_path)

    input_files = sorted(
        {str(candidate) for candidate in candidates if candidate.suffix.lower() in SUPPORTED_INPUT_SUFFIXES}
    )
    parquet_count = sum(file_name.lower().endswith('.parquet') for file_name in input_files)
    jsonl_count = sum(file_name.lower().endswith('.jsonl') for file_name in input_files)
    print('Discovered input files:')
    print(f'  Parquet: {parquet_count}')
    print(f'  JSONL: {jsonl_count}')
    print(f'  Total: {len(input_files)}')

    if not input_files:
        raise ValueError(
            f"No .parquet or .jsonl files found for input {input_path!r}. "
            "Directory inputs are searched recursively."
        )
    return input_files


def get_parquet_module():
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Parquet preprocessing requires pyarrow. Install pyarrow or provide JSONL input."
        ) from error
    return parquet


def validate_parquet_inputs(input_files, json_keys):
    parquet_files = [file_name for file_name in input_files if file_name.lower().endswith('.parquet')]
    if not parquet_files:
        return

    parquet = get_parquet_module()
    for input_file_name in parquet_files:
        schema_names = parquet.ParquetFile(input_file_name).schema.names
        missing_keys = sorted(set(json_keys) - set(schema_names))
        if missing_keys:
            raise ValueError(
                f"Parquet file {input_file_name!r} is missing required columns: {missing_keys}"
            )


def count_jsonl_documents(input_files):
    total = 0
    for input_file_name in input_files:
        if input_file_name.lower().endswith(('.jsonl', '.jsonl.tmp')):
            with open(input_file_name, 'rb') as fin:
                total += sum(1 for _ in fin)
    return total


def count_input_documents(input_files):
    """Return exact document counts without reading parquet row contents."""
    parquet_files = [file_name for file_name in input_files if file_name.lower().endswith('.parquet')]
    jsonl_files = [file_name for file_name in input_files if file_name.lower().endswith('.jsonl')]

    parquet_documents = 0
    if parquet_files:
        parquet = get_parquet_module()
        parquet_documents = sum(parquet.ParquetFile(file_name).metadata.num_rows for file_name in parquet_files)
    jsonl_documents = count_jsonl_documents(jsonl_files)
    total_documents = parquet_documents + jsonl_documents

    print('Input documents:')
    print(f'  Parquet: {parquet_documents}')
    print(f'  JSONL: {jsonl_documents}')
    print(f'  Total: {total_documents}')
    if total_documents == 0:
        raise ValueError('Input files contain no documents.')
    return total_documents


def format_duration(seconds):
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def iter_input_records(input_files, json_keys, parquet_batch_size=1024):
    """Yield JSON lines or parquet rows from a deterministic list of files."""
    for input_file_name in input_files:
        if input_file_name.lower().endswith(('.jsonl', '.jsonl.tmp')):
            with open(input_file_name, 'r', encoding='utf-8') as fin:
                yield from fin
            continue

        parquet = get_parquet_module()
        parquet_file = parquet.ParquetFile(input_file_name)
        missing_keys = sorted(set(json_keys) - set(parquet_file.schema.names))
        if missing_keys:
            raise ValueError(
                f"Parquet file {input_file_name!r} is missing required columns: {missing_keys}"
            )
        for batch in parquet_file.iter_batches(columns=json_keys, batch_size=parquet_batch_size):
            yield from batch.to_pylist()


def record_size(record, json_keys):
    if isinstance(record, str):
        return len(record.encode('utf-8'))
    return sum(len(str(record[key]).encode('utf-8')) for key in json_keys)


def record_to_json_line(record):
    if isinstance(record, str):
        return record if record.endswith('\n') else record + '\n'
    return json.dumps(record) + '\n'


def collect_process_results(processes, result_queue):
    """Collect one result per process and fail if a child exits without reporting."""
    results = []
    pending_processes = {process.pid: process for process in processes}
    while pending_processes:
        try:
            status, process_id, payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            silent = [
                process
                for process in pending_processes.values()
                if process.exitcode is not None
            ]
            if not silent:
                continue
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            silent_pids = ", ".join(str(process.pid) for process in silent)
            raise RuntimeError(
                f"Preprocessing worker process(es) {silent_pids} exited without returning a result"
            )

        if process_id not in pending_processes:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            raise RuntimeError(
                "Received an unexpected or duplicate result from preprocessing "
                f"process {process_id}"
            )
        pending_processes.pop(process_id)
        if status == "error":
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            raise RuntimeError(f"Preprocessing worker failed:\n{payload}")
        results.append(payload)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Preprocessing worker process {process.pid} exited with code {process.exitcode}"
            )
    return results


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, record):
        data = json.loads(record) if isinstance(record, str) else record
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), record_size(record, self.args.json_keys)

    def encode(self, record):
        data = json.loads(record) if isinstance(record, str) else record
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, record_size(record, self.args.json_keys)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers
        self.performance = []
        self.last_log_time = None
        self.last_log_count = 0
        self.smoothed_docs_per_second = None

    def print_processing_stats(
        self, count, proc_start, total_bytes_processed, total_documents=None, force=False
    ):
        if force and self.last_log_time is not None and self.last_log_count == count:
            return
        completed = force or (total_documents is not None and count >= total_documents)
        if count % self.args.log_interval != 0 and not completed:
            return

        current = time.monotonic()
        if (
            self.last_log_time is not None
            and not completed
            and current - self.last_log_time < self.args.log_interval_seconds
        ):
            return

        elapsed = max(current - proc_start, 1e-9)
        average_docs_per_second = count / elapsed
        if self.last_log_time is None:
            interval_docs_per_second = average_docs_per_second
        else:
            interval_elapsed = max(current - self.last_log_time, 1e-9)
            interval_docs_per_second = (count - self.last_log_count) / interval_elapsed
        if self.smoothed_docs_per_second is None:
            self.smoothed_docs_per_second = interval_docs_per_second
        else:
            self.smoothed_docs_per_second = (
                0.2 * interval_docs_per_second + 0.8 * self.smoothed_docs_per_second
            )

        mbs = total_bytes_processed / elapsed / 1024 / 1024
        if total_documents is None:
            progress = f"Processed {count:,} documents"
            eta = "unknown"
        else:
            percentage = min(100.0, 100.0 * count / total_documents)
            progress = f"Processed {count:,}/{total_documents:,} documents ({percentage:.2f}%)"
            remaining = max(0, total_documents - count)
            eta_seconds = remaining / max(self.smoothed_docs_per_second, 1e-9)
            eta = format_duration(eta_seconds)

        print(
            f"{progress} | {self.smoothed_docs_per_second:,.1f} docs/s | "
            f"{mbs:.2f} MB/s | elapsed {format_duration(elapsed)} | ETA {eta}",
            file=sys.stderr,
        )
        self.last_log_time = current
        self.last_log_count = count
        if self.args.find_optimal_num_workers:
            self.performance.append(average_docs_per_second)

    def split_sentences(self, file_name):
        input_file_names, output_file_name, total_documents = file_name
        records = iter_input_records(
            input_file_names, self.args.json_keys, self.args.parquet_batch_size
        )
        fout = open(output_file_name, 'w', encoding='utf-8')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, records, 32)

        proc_start = time.monotonic()
        total_bytes_processed = 0
        processed_documents = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            processed_documents = i
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed, total_documents)
        self.print_processing_stats(
            processed_documents,
            proc_start,
            total_bytes_processed,
            total_documents,
            force=True,
        )

        fout.close()

        pool.close()
        pool.join()

    def process_json_file(self, file_name):
        input_file_names, output_prefix, total_documents = file_name
        records = iter_input_records(
            input_file_names, self.args.json_keys, self.args.parquet_batch_size
        )

        startup_start = time.monotonic()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, records, 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.monotonic()
        proc_start = time.monotonic()
        total_bytes_processed = 0
        processed_documents = 0
        total_tokens = {key: 0 for key in self.args.json_keys}
        print("Time to startup:", startup_end - startup_start)
        if self.args.find_optimal_num_workers and total_documents is not None:
            total_documents = min(total_documents, self.args.max_documents)
        try:
            for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
                if self.args.find_optimal_num_workers and i > self.args.max_documents:
                    break
                processed_documents = i
                total_bytes_processed += bytes_processed
                for key in doc.keys():
                    builders[key].add_document(doc[key], sentence_lens[key])
                    total_tokens[key] += len(doc[key])
                self.print_processing_stats(
                    i, proc_start, total_bytes_processed, total_documents
                )
            self.print_processing_stats(
                processed_documents,
                proc_start,
                total_bytes_processed,
                total_documents,
                force=True,
            )

            empty_keys = [key for key, token_count in total_tokens.items() if token_count == 0]
            if empty_keys:
                raise ValueError(
                    f"Tokenization produced zero tokens for JSON key(s) {empty_keys} from "
                    f"{input_file_names!r}"
                )

            for key in self.args.json_keys:
                builders[key].finalize(output_idx_files[key])
        except BaseException:
            pool.terminate()
            pool.join()
            for builder in builders.values():
                if not builder.data_file.closed:
                    builder.data_file.close()
            for output_path in list(output_bin_files.values()) + list(output_idx_files.values()):
                if os.path.exists(output_path):
                    os.remove(output_path)
            raise
        else:
            pool.close()
            pool.join()

        return self.performance


def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help=('Input .jsonl/.parquet file, glob, or directory. '
                             'Directories are searched recursively.'))
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group.add_argument('--parquet-batch-size', type=int, default=1024,
                       help='Rows read from parquet at a time (default: 1024).')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--find-optimal-num-workers', action='store_true',
                       help=('Find optimal number of workers.'
                             'Script will run few small jobs with '
                             'different number of workers to define '
                             'optimal number of workers in terms of performance.'))
    group.add_argument('--workers-to-check', nargs='+', type=int, default=[16, 32, 64],
                       help=('list of workers to run data processing with '
                             'to find optimal number of workers. '
                             'Works only when --find-optimal-num-workers is enabled. '))
    group.add_argument('--max-documents', type=int, default=100_000,
                       help=('Maximum number of documents to preprocess '
                             'to find  optimal number of workers.'
                             'Works only when --find-optimal-num-workers is enabled.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--log-interval-seconds', type=float, default=30.0,
                       help='Minimum seconds between progress updates (default: 30).')
    group.add_argument('--skip-document-count', action='store_true',
                       help='Skip exact input counting; percentage and ETA will be unavailable.')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    if args.parquet_batch_size < 1:
        parser.error('--parquet-batch-size must be at least 1')
    if args.log_interval < 1:
        parser.error('--log-interval must be at least 1')
    if args.log_interval_seconds < 0:
        parser.error('--log-interval-seconds cannot be negative')
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    input_file_name = args.output_prefix + "_partition_" + str(file_id) + ".jsonl.tmp"
    sentence_split_file = args.output_prefix + "_ss_" + str(file_id) + ".jsonl.tmp"
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        paths = in_ss_out_names[i][key]
        if isinstance(paths, str):
            paths = [paths]
        if not all(os.path.exists(path) for path in paths):
            return False
    return True


def find_optimal_num_workers(performance, partitions):
    """Parses saved .json files with perf. numbers and prints optimal number of workers"""
    results = []

    # each file assumed to contain a single {workers: [perf_list]}
    for workers, perf_list in performance.items():
        workers = int(workers)
        avg_perf = np.mean(perf_list)
        results.append((workers, avg_perf))

    # sort by average performance (descending: fastest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n-----------------------------------")
    print("Performance results (fastest → slowest):")
    for i, (workers, avg_perf) in enumerate(results):
        print(f"{i+1}. {workers * partitions} workers → avg. docs/s: {avg_perf:.4f}")
    
    best_workers, best_perf = results[0]

    print("\n-----------------------------------")
    print(
        f"The most optimal num of workers is {best_workers * partitions} "
        f"with avg. preprocessed docs/s: {best_perf:.4f}."
    )
    print("-----------------------------------")


def main():
    args = get_args()
    input_files = discover_input_files(args.input)
    validate_parquet_inputs(input_files, args.json_keys)
    if args.skip_document_count:
        total_documents = None
        print('Document counting skipped; percentage and ETA will be unavailable.')
    else:
        print('Counting input documents for progress and ETA...')
        total_documents = count_input_documents(input_files)

    workers = args.workers_to_check if args.find_optimal_num_workers else [args.workers]
    for num_workers in workers:
        if num_workers % args.partitions != 0:
            print(
                f"Removing num_workers ({num_workers}) from workers list "
                f"because it's not divisible by num_partitions ({args.partitions})"
            )
            workers.remove(num_workers)
    assert workers, "Please, provide valid number of workers which is divisible by number of partitions."
    if args.find_optimal_num_workers:
        args.log_interval = 1000

    performance = {}
    for num_workers in workers:
        print(f"Processing data with {num_workers} workers.")
        if args.split_sentences:
            if nltk_available:
                nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
            else:
                raise Exception(
                    "nltk library required for sentence splitting is not available.")

        in_ss_out_names = []
        if args.partitions == 1:
            sentence_split_file = args.output_prefix + "_ss.jsonl.tmp"
            file_names = {
                'partition': input_files,
                'sentence_split': sentence_split_file,
                'output_prefix': args.output_prefix,
                'num_documents': total_documents}
            in_ss_out_names.append(file_names)
        else:
            # Count total records across every supported input file.
            if args.keep_sequential_samples:
                total_sample_count = total_documents
                if total_sample_count is None:
                    total_sample_count = sum(
                        1
                        for _ in iter_input_records(
                            input_files, args.json_keys, args.parquet_batch_size
                        )
                    )
                partition_size = math.ceil(total_sample_count / args.partitions)

            # create .jsonl parition files
            for idx in range(args.partitions):
                in_ss_out_name = get_file_name(args, idx)
                in_ss_out_names.append(in_ss_out_name)

            # check to see if paritions were already created
            partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

            # check to see if paritions with split sentences already created
            split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

            if not partitions_present and not split_sentences_present:
                # populate .jsonl partition files from parent files
                partitioned_input_files = []
                partition_document_counts = [0] * args.partitions
                for idx in range(args.partitions):
                    partitioned_input_file = open(
                        in_ss_out_names[idx]['partition'], 'w', encoding='utf-8'
                    )
                    partitioned_input_files.append(partitioned_input_file)

                index = 0
                if args.keep_sequential_samples: line_count = 0
                for record in iter_input_records(
                    input_files, args.json_keys, args.parquet_batch_size
                ):
                    partitioned_input_files[index].write(record_to_json_line(record))
                    partition_document_counts[index] += 1
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index = min(index + 1, args.partitions - 1)
                    else:
                        index = (index + 1) % args.partitions

                for idx in range(args.partitions):
                    partitioned_input_files[idx].close()
                    in_ss_out_names[idx]['num_documents'] = partition_document_counts[idx]
            elif not args.skip_document_count:
                for name in in_ss_out_names:
                    name['num_documents'] = count_jsonl_documents([name['partition']])
            else:
                for name in in_ss_out_names:
                    name['num_documents'] = None

        partition = Partition(args, num_workers//args.partitions)

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        # split sentences in partition files
        if args.split_sentences and not split_sentences_present:
            processes = []
            for name in in_ss_out_names:
                p = multiprocessing.Process(target=partition.split_sentences,
                                            args=((name['partition'] if isinstance(name['partition'], list)
                                                   else [name['partition']],
                                                   name['sentence_split'],
                                                   name['num_documents']),))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Sentence-splitting worker process {p.pid} exited with code {p.exitcode}"
                    )

            if args.partitions == 1:
                continue

        def process_json_file(name, q, input_key):
            try:
                input_file_names = name[input_key]
                if isinstance(input_file_names, str):
                    input_file_names = [input_file_names]
                worker_performance = partition.process_json_file(
                    (input_file_names, name['output_prefix'], name['num_documents'])
                )
                q.put(("ok", os.getpid(), worker_performance))
            except BaseException:
                q.put(("error", os.getpid(), traceback.format_exc()))
                raise

        # encode partition files in parallel
        processes = []
        input_key = 'sentence_split' if args.split_sentences else 'partition'
        q = multiprocessing.Queue()
        for name in in_ss_out_names:
            p = multiprocessing.Process(target=process_json_file, args=(name, q, input_key))

            p.start()
            processes.append(p)

        worker_results = collect_process_results(processes, q)
        for worker_performance in worker_results:
            if args.find_optimal_num_workers:
                performance[num_workers] = worker_performance

        if args.partitions == 1:
            continue

        # merge bin/idx partitions
        level = "document"
        if args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        tokenizer = build_tokenizer(args)

        for key in args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                        key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                        key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

            for name in in_ss_out_names:
                parition_output_prefix = name['output_prefix']
                full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                                key, level)
                builders[key].add_index(full_partition_output_prefix)
            builders[key].finalize(output_idx_files[key])

    # Find the most optimal number of workers
    if args.find_optimal_num_workers:
        find_optimal_num_workers(performance, args.partitions)

if __name__ == '__main__':

    main()
