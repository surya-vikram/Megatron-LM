# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipelined preprocessing with dedicated readers, tokenizers, and shard writers."""

import argparse
import glob
import json
import multiprocessing
import os
import queue
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from megatron.core.datasets import indexed_dataset
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args

SUPPORTED_INPUT_SUFFIXES = {".jsonl", ".parquet"}


class ByteBudget:
    """Cross-process byte credits for a multiprocessing queue."""

    def __init__(self, context, limit):
        self.limit = max(1, int(limit))
        self.used = context.Value("Q", 0)
        self.condition = context.Condition()

    def acquire(self, amount):
        amount = max(1, int(amount))
        with self.condition:
            while self.used.value and self.used.value + amount > self.limit:
                self.condition.wait()
            self.used.value += amount
        return amount

    def release(self, amount):
        with self.condition:
            self.used.value -= int(amount)
            self.condition.notify_all()


def discover_input_files(input_path):
    path = Path(input_path)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    else:
        candidates = []
        for match in glob.glob(input_path, recursive=True):
            match_path = Path(match)
            if match_path.is_dir():
                candidates.extend(candidate for candidate in match_path.rglob("*") if candidate.is_file())
            elif match_path.is_file():
                candidates.append(match_path)
    files = sorted(
        {str(candidate) for candidate in candidates if candidate.suffix.lower() in SUPPORTED_INPUT_SUFFIXES}
    )
    if not files:
        raise ValueError(f"No .jsonl or .parquet inputs found for {input_path!r}")
    return files


def get_parquet_module():
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Parquet preprocessing requires pyarrow") from error
    return parquet


def build_input_tasks(input_files, json_keys, jsonl_chunk_bytes):
    tasks = []
    total_input_bytes = 0
    task_id = 0
    parquet = None
    for file_id, path in enumerate(input_files):
        file_size = os.path.getsize(path)
        total_input_bytes += file_size
        if path.lower().endswith(".jsonl"):
            for start in range(0, max(file_size, 1), jsonl_chunk_bytes):
                tasks.append((task_id, file_id, "jsonl", path, start, min(file_size, start + jsonl_chunk_bytes)))
                task_id += 1
            continue

        parquet = parquet or get_parquet_module()
        parquet_file = parquet.ParquetFile(path)
        missing = sorted(set(json_keys) - set(parquet_file.schema.names))
        if missing:
            raise ValueError(f"Parquet file {path!r} is missing columns {missing}")
        for row_group in range(parquet_file.num_row_groups):
            tasks.append((task_id, file_id, "parquet", path, row_group, row_group + 1))
            task_id += 1
    return tasks, total_input_bytes


def iter_jsonl_range(path, start, end, json_keys):
    with open(path, "rb") as stream:
        if start:
            stream.seek(start - 1)
            if stream.read(1) != b"\n":
                stream.readline()
        else:
            stream.seek(0)

        while stream.tell() < end:
            line = stream.readline()
            if not line:
                break
            data = json.loads(line)
            yield tuple(data[key] for key in json_keys), len(line)


def iter_parquet_row_group(path, row_group, json_keys):
    parquet = get_parquet_module()
    table = parquet.ParquetFile(path).read_row_group(
        row_group, columns=json_keys, use_threads=False
    )
    columns = [table.column(key).to_pylist() for key in json_keys]
    for values in zip(*columns):
        record_bytes = sum(len(str(value).encode("utf-8")) for value in values)
        yield values, record_bytes


def put_raw_batch(raw_queue, raw_budget, task, part_id, records, input_bytes, counters):
    estimated_bytes = input_bytes + len(records) * 64
    reserved = raw_budget.acquire(estimated_bytes)
    raw_queue.put((task[0], task[1], part_id, records, input_bytes, reserved))
    with counters["read_documents"].get_lock():
        counters["read_documents"].value += len(records)
    with counters["read_bytes"].get_lock():
        counters["read_bytes"].value += input_bytes


def reader_worker(args, task_queue, raw_queue, raw_budget, counters, errors):
    try:
        while True:
            task = task_queue.get()
            if task is None:
                return
            task_id, _, kind, path, start, end = task
            del task_id
            if kind == "jsonl":
                records = iter_jsonl_range(path, start, end, args.json_keys)
            else:
                records = iter_parquet_row_group(path, start, args.json_keys)

            batch = []
            batch_bytes = 0
            part_id = 0
            for record, record_bytes in records:
                if batch and batch_bytes + record_bytes > args.raw_batch_bytes:
                    put_raw_batch(
                        raw_queue, raw_budget, task, part_id, batch, batch_bytes, counters
                    )
                    batch = []
                    batch_bytes = 0
                    part_id += 1
                batch.append(record)
                batch_bytes += record_bytes
            if batch:
                put_raw_batch(raw_queue, raw_budget, task, part_id, batch, batch_bytes, counters)
    except BaseException:
        errors.put(("reader", os.getpid(), traceback.format_exc()))
        raise


def initialize_tokenizer(args):
    initialize_tokenizer.tokenizer = build_tokenizer(args)


def encode_batch(args, records, dtype):
    payloads = {}
    all_lengths = {}
    document_sequence_counts = {}
    total_tokens = 0
    for key_index, key in enumerate(args.json_keys):
        token_ids = []
        sequence_lengths = []
        per_document_counts = []
        for values in records:
            text = values[key_index]
            sentences = text if isinstance(text, list) else [text]
            document_ids = []
            document_lengths = []
            for sentence in sentences:
                sentence_ids = initialize_tokenizer.tokenizer.tokenize(sentence)
                if sentence_ids:
                    document_ids.extend(sentence_ids)
                    document_lengths.append(len(sentence_ids))
            if document_ids and args.append_eod:
                document_ids.append(initialize_tokenizer.tokenizer.eod)
                document_lengths[-1] += 1
            token_ids.extend(document_ids)
            sequence_lengths.extend(document_lengths)
            per_document_counts.append(len(document_lengths))
        payloads[key] = np.asarray(token_ids, dtype=dtype).tobytes(order="C")
        all_lengths[key] = sequence_lengths
        document_sequence_counts[key] = per_document_counts
        total_tokens += len(token_ids)
    return payloads, all_lengths, document_sequence_counts, total_tokens


def tokenizer_worker(args, dtype_string, raw_queue, raw_budget, encoded_queue, encoded_budget,
                     counters, errors):
    try:
        initialize_tokenizer(args)
        dtype = np.dtype(dtype_string)
        while True:
            item = raw_queue.get()
            if item is None:
                return
            task_id, file_id, part_id, records, input_bytes, raw_reserved = item
            payloads, lengths, document_counts, total_tokens = encode_batch(args, records, dtype)
            document_count = len(records)
            del records
            raw_budget.release(raw_reserved)

            encoded_bytes = (
                sum(len(payload) for payload in payloads.values())
                + sum(len(value) * 4 for value in lengths.values())
                + sum(len(value) * 4 for value in document_counts.values())
            )
            encoded_reserved = encoded_budget.acquire(encoded_bytes)
            encoded_queue.put(
                (
                    task_id,
                    file_id,
                    part_id,
                    payloads,
                    lengths,
                    document_counts,
                    document_count,
                    total_tokens,
                    input_bytes,
                    encoded_reserved,
                )
            )
            with counters["tokenized_documents"].get_lock():
                counters["tokenized_documents"].value += document_count
            with counters["tokenized_tokens"].get_lock():
                counters["tokenized_tokens"].value += total_tokens
    except BaseException:
        errors.put(("tokenizer", os.getpid(), traceback.format_exc()))
        raise


def add_encoded_batch(builder, payload, lengths, document_counts):
    builder.data_file.write(payload)
    builder.sequence_lengths.extend(lengths)
    sequence_index = builder.document_indices[-1]
    for count in document_counts:
        sequence_index += count
        builder.document_indices.append(sequence_index)


def writer_worker(args, writer_id, dtype_string, level, work_dir, encoded_queue,
                  encoded_budget, counters, errors):
    builders = {}
    try:
        dtype = np.dtype(dtype_string).type
        shard_prefix = os.path.join(work_dir, f"shard-{writer_id:05d}")
        for key in args.json_keys:
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                f"{shard_prefix}_{key}_{level}.bin", dtype=dtype
            )

        while True:
            item = encoded_queue.get()
            if item is None:
                break
            (
                _,
                _,
                _,
                payloads,
                lengths,
                document_counts,
                document_count,
                total_tokens,
                _,
                encoded_reserved,
            ) = item
            for key in args.json_keys:
                add_encoded_batch(
                    builders[key], payloads[key], lengths[key], document_counts[key]
                )
            encoded_budget.release(encoded_reserved)
            with counters["written_documents"].get_lock():
                counters["written_documents"].value += document_count
            with counters["written_tokens"].get_lock():
                counters["written_tokens"].value += total_tokens

        for key in args.json_keys:
            builders[key].finalize(f"{shard_prefix}_{key}_{level}.idx")
    except BaseException:
        for builder in builders.values():
            if not builder.data_file.closed:
                builder.data_file.close()
        errors.put(("writer", os.getpid(), traceback.format_exc()))
        raise


def queue_size(work_queue):
    try:
        return work_queue.qsize()
    except (NotImplementedError, OSError):
        return -1


def raise_process_error(processes, errors):
    try:
        stage, pid, detail = errors.get_nowait()
    except queue.Empty:
        stage = pid = detail = None
    if detail is not None:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(f"{stage} process {pid} failed:\n{detail}")
    failed = [process for process in processes if process.exitcode not in (None, 0)]
    if failed:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(
            "Pipeline worker(s) failed: "
            + ", ".join(f"pid={process.pid} exit={process.exitcode}" for process in failed)
        )


def wait_for_processes(targets, all_processes, errors, counters, raw_queue, encoded_queue,
                       started, log_interval_seconds):
    next_log = time.monotonic() + log_interval_seconds
    while any(process.exitcode is None for process in targets):
        raise_process_error(all_processes, errors)
        now = time.monotonic()
        if now >= next_log:
            elapsed = max(now - started, 1e-9)
            print(
                "Pipeline | "
                f"read={counters['read_documents'].value:,} docs | "
                f"tokenized={counters['tokenized_documents'].value:,} docs | "
                f"written={counters['written_documents'].value:,} docs | "
                f"tokens={counters['written_tokens'].value:,} | "
                f"raw_q={queue_size(raw_queue)} encoded_q={queue_size(encoded_queue)} | "
                f"{counters['written_tokens'].value / elapsed:,.0f} tokens/s",
                file=sys.stderr,
            )
            next_log = now + log_interval_seconds
        time.sleep(0.1)
    for process in targets:
        process.join()
    raise_process_error(all_processes, errors)


def merge_shards(args, dtype, level, work_dir, writer_count):
    merge_started = time.monotonic()
    for key in args.json_keys:
        output_prefix = f"{args.output_prefix}_{key}_{level}"
        builder = indexed_dataset.IndexedDatasetBuilder(output_prefix + ".bin", dtype=dtype)
        for writer_id in range(writer_count):
            shard_prefix = os.path.join(work_dir, f"shard-{writer_id:05d}_{key}_{level}")
            builder.add_index(shard_prefix)
        builder.finalize(output_prefix + ".idx")
    return time.monotonic() - merge_started


def build_counters(context):
    return {
        "read_documents": context.Value("Q", 0),
        "read_bytes": context.Value("Q", 0),
        "tokenized_documents": context.Value("Q", 0),
        "tokenized_tokens": context.Value("Q", 0),
        "written_documents": context.Value("Q", 0),
        "written_tokens": context.Value("Q", 0),
    }


def get_args():
    parser = _add_tokenizer_args(argparse.ArgumentParser())
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--json-keys", nargs="+", default=["text"])
    parser.add_argument("--append-eod", action="store_true")
    parser.add_argument("--num-readers", type=int, required=True)
    parser.add_argument("--num-tokenizers", type=int, required=True)
    parser.add_argument("--num-writers", type=int, required=True)
    parser.add_argument("--queue-memory-budget-gb", type=float, required=True)
    parser.add_argument("--raw-batch-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--jsonl-chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()
    if args.num_readers < 1:
        parser.error("--num-readers must be at least 1")
    if args.num_tokenizers < 1:
        parser.error("--num-tokenizers must be at least 1")
    if args.num_writers < 1:
        parser.error("--num-writers must be at least 1")
    if args.queue_memory_budget_gb <= 0:
        parser.error("--queue-memory-budget-gb must be positive")
    args.keep_empty = False
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    return args


def main():
    args = get_args()
    reader_count = args.num_readers
    writer_count = args.num_writers
    process_count = reader_count + args.num_tokenizers + writer_count

    level = "document"
    input_files = discover_input_files(args.input)
    tasks, total_input_bytes = build_input_tasks(
        input_files, args.json_keys, args.jsonl_chunk_bytes
    )
    tokenizer = build_tokenizer(args)
    dtype = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
    dtype_string = np.dtype(dtype).str
    del tokenizer

    work_dir = args.output_prefix + ".pipeline_shards"
    if os.path.exists(work_dir):
        raise FileExistsError(f"Pipeline work directory already exists: {work_dir}")
    for key in args.json_keys:
        for suffix in (".bin", ".idx"):
            output_path = f"{args.output_prefix}_{key}_{level}{suffix}"
            if os.path.exists(output_path):
                raise FileExistsError(f"Output already exists: {output_path}")
    os.makedirs(work_dir)

    context = multiprocessing.get_context("fork")
    task_queue = context.Queue()
    queue_items = max(32, process_count * 2)
    raw_queue = context.Queue(maxsize=queue_items)
    encoded_queue = context.Queue(maxsize=queue_items)
    errors = context.Queue()
    counters = build_counters(context)
    queue_budget = int(args.queue_memory_budget_gb * 1024**3)
    raw_budget = ByteBudget(context, queue_budget // 2)
    encoded_budget = ByteBudget(context, queue_budget // 2)

    readers = [
        context.Process(
            target=reader_worker,
            args=(args, task_queue, raw_queue, raw_budget, counters, errors),
            name=f"reader-{index}",
        )
        for index in range(reader_count)
    ]
    tokenizers = [
        context.Process(
            target=tokenizer_worker,
            args=(
                args,
                dtype_string,
                raw_queue,
                raw_budget,
                encoded_queue,
                encoded_budget,
                counters,
                errors,
            ),
            name=f"tokenizer-{index}",
        )
        for index in range(args.num_tokenizers)
    ]
    writers = [
        context.Process(
            target=writer_worker,
            args=(
                args,
                index,
                dtype_string,
                level,
                work_dir,
                encoded_queue,
                encoded_budget,
                counters,
                errors,
            ),
            name=f"writer-{index}",
        )
        for index in range(writer_count)
    ]
    all_processes = readers + tokenizers + writers

    print("Pipeline configuration:")
    print(f"  Input files: {len(input_files)}")
    print(f"  Input tasks: {len(tasks)}")
    print(f"  Input bytes: {total_input_bytes:,}")
    print(f"  Readers: {reader_count}")
    print(f"  Tokenizers: {args.num_tokenizers}")
    print(f"  Writers/shards: {writer_count}")
    print(f"  Total worker processes: {process_count}")
    print(f"  Queue byte budget: {queue_budget:,}")

    started = time.monotonic()
    try:
        for process in all_processes:
            process.start()
        for task in tasks:
            task_queue.put(task)
        for _ in readers:
            task_queue.put(None)

        wait_for_processes(
            readers,
            all_processes,
            errors,
            counters,
            raw_queue,
            encoded_queue,
            started,
            args.log_interval_seconds,
        )
        for _ in tokenizers:
            raw_queue.put(None)
        wait_for_processes(
            tokenizers,
            all_processes,
            errors,
            counters,
            raw_queue,
            encoded_queue,
            started,
            args.log_interval_seconds,
        )
        for _ in writers:
            encoded_queue.put(None)
        wait_for_processes(
            writers,
            all_processes,
            errors,
            counters,
            raw_queue,
            encoded_queue,
            started,
            args.log_interval_seconds,
        )

        processing_seconds = time.monotonic() - started
        if not (
            counters["read_documents"].value
            == counters["tokenized_documents"].value
            == counters["written_documents"].value
        ):
            raise RuntimeError("Reader/tokenizer/writer document counts do not match")
        if counters["tokenized_tokens"].value != counters["written_tokens"].value:
            raise RuntimeError("Tokenizer/writer token counts do not match")
        if counters["written_tokens"].value == 0:
            raise ValueError("Tokenization produced zero tokens")

        merge_seconds = merge_shards(args, dtype, level, work_dir, writer_count)
        total_seconds = time.monotonic() - started
        summary = {
            "documents": counters["written_documents"].value,
            "tokens": counters["written_tokens"].value,
            "input_text_bytes": counters["read_bytes"].value,
            "processing_seconds": processing_seconds,
            "merge_seconds": merge_seconds,
            "total_seconds": total_seconds,
            "documents_per_second": counters["written_documents"].value / total_seconds,
            "tokens_per_second": counters["written_tokens"].value / total_seconds,
            "readers": reader_count,
            "tokenizers": args.num_tokenizers,
            "writers": writer_count,
        }
        print("PIPELINE_SUMMARY " + json.dumps(summary, sort_keys=True))
        if not args.keep_shards:
            shutil.rmtree(work_dir)
    except BaseException:
        for process in all_processes:
            if process.is_alive():
                process.terminate()
        for process in all_processes:
            process.join()
        raise


if __name__ == "__main__":
    main()
