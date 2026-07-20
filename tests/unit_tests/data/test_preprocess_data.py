# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import os
import runpy
import sys
import tempfile

import nltk
import pytest
import requests

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer import MEGATRON_CONFIG_MAP
from tools.merge_datasets import main as merge_main
from tools.preprocess_data import Encoder
from tools.preprocess_data import Partition
from tools.preprocess_data import count_input_documents
from tools.preprocess_data import discover_input_files
from tools.preprocess_data import format_duration
from tools.preprocess_data import get_args as build_args
from tools.preprocess_data import iter_input_records
from tools.preprocess_data import main as build_main
from tools.preprocess_data import validate_parquet_inputs

__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB = (
    "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
)

__LOCAL_BERT_VOCAB = "/home/gitlab-runner/data/bert_data/vocab.txt"

__LOCAL_GPT2_MERGE = "/home/gitlab-runner/data/gpt3_data/gpt2-merges.txt"

__LOCAL_GPT2_VOCAB = "/home/gitlab-runner/data/gpt3_data/gpt2-vocab.json"


def dummy_jsonl(odir):
    # numbers
    list_numbers = [json.dumps({"text": str(i + 1)}) + "\n" for i in range(100)]
    with open(os.path.join(odir, "numbers.jsonl"), "w") as writer:
        writer.writelines(list_numbers)
    # numbers ascending
    list_numbers_ascending = [
        json.dumps({"text": " ".join([str(j + 1) for j in range(i + 1)])}) + "\n"
        for i in range(100)
    ]
    with open(os.path.join(odir, "numbers_ascending.jsonl"), "w") as writer:
        writer.writelines(list_numbers_ascending)
    # test
    list_test = []
    with open(__file__) as reader:
        for line in reader:
            list_test.append(json.dumps({"text": line}) + "\n")
    with open(os.path.join(odir, "test.jsonl"), "w") as writer:
        writer.writelines(list_test)


def test_recursive_jsonl_and_parquet_discovery(tmp_path, capsys):
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")

    jsonl_path = tmp_path / "nested" / "records.jsonl"
    jsonl_path.parent.mkdir()
    jsonl_path.write_text(json.dumps({"text": "from jsonl"}) + "\n", encoding="utf-8")

    parquet_path = tmp_path / "deeper" / "nested" / "records.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet.write_table(pyarrow.table({"text": ["from parquet"]}), parquet_path)

    (tmp_path / "ignored.json").write_text('{"text": "ignored"}\n', encoding="utf-8")
    (tmp_path / "partial.parquet.incomplete").write_bytes(b"ignored")

    input_files = discover_input_files(str(tmp_path))
    assert input_files == sorted([str(parquet_path), str(jsonl_path)])
    assert capsys.readouterr().out == (
        "Discovered input files:\n"
        "  Parquet: 1\n"
        "  JSONL: 1\n"
        "  Total: 2\n"
    )

    records = list(iter_input_records(input_files, ["text"], parquet_batch_size=1))
    normalized = [json.loads(record) if isinstance(record, str) else record for record in records]
    assert normalized == [{"text": "from parquet"}, {"text": "from jsonl"}]

    assert count_input_documents(input_files) == 2
    assert capsys.readouterr().out == (
        "Opening " + str(parquet_path) + "\n"
        "Opening " + str(jsonl_path) + "\n"
        "Input documents:\n"
        "  Parquet: 1\n"
        "  JSONL: 1\n"
        "  Total: 2\n"
    )


def test_parquet_requires_json_keys(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")

    parquet_path = tmp_path / "records.parquet"
    parquet.write_table(pyarrow.table({"content": ["wrong column"]}), parquet_path)

    with pytest.raises(ValueError, match="missing required columns: \\['text'\\]"):
        validate_parquet_inputs([str(parquet_path)], ["text"])


def test_progress_includes_percentage_and_eta(monkeypatch, capsys):
    args = type(
        "Args",
        (),
        {
            "find_optimal_num_workers": False,
            "log_interval": 1,
            "log_interval_seconds": 30,
        },
    )()
    partition = Partition(args, workers=1)
    monkeypatch.setattr("tools.preprocess_data.time.monotonic", lambda: 10.0)

    partition.print_processing_stats(
        count=50,
        proc_start=0.0,
        total_bytes_processed=100 * 1024 * 1024,
        total_documents=100,
    )

    assert capsys.readouterr().err == (
        "Processed 50/100 documents (50.00%) | 5.0 docs/s | 10.00 MB/s | "
        "elapsed 00:00:10 | ETA 00:00:10\n"
    )
    assert format_duration(3661.9) == "01:01:01"


def build_datasets(idir, odir, extra_args=[]):
    for name in os.listdir(idir):
        sys.argv = [
            sys.argv[0],
            "--input",
            os.path.join(idir, name),
            "--output-prefix",
            os.path.join(odir, os.path.splitext(name)[0]),
        ] + extra_args
        build_main()


def merge_datasets(idir):
    sys.argv = [sys.argv[0], "--input", idir, "--output-prefix", os.path.join(idir, "merge")]
    merge_main()


def do_test_preprocess_data(temp_dir, extra_args=[]):
    # set the default nltk data path
    os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
    nltk.data.path.append(os.environ["NLTK_DATA"])

    path_to_raws = os.path.join(temp_dir, "sample_raws")
    path_to_data = os.path.join(temp_dir, "sample_data")
    os.mkdir(path_to_raws)
    os.mkdir(path_to_data)

    # create the dummy resources
    dummy_jsonl(path_to_raws)

    # build the datasets
    build_datasets(path_to_raws, path_to_data, extra_args=extra_args)

    # merge the datasets
    merge_datasets(path_to_data)

    sys.argv = [sys.argv[0], "--input", None, "--output-prefix", None] + extra_args
    encoder = Encoder(build_args())
    encoder.initializer()

    def tokens_to_string(toks):
        for option in ["decode", "detokenize"]:
            try:
                return getattr(encoder.tokenizer, option)(toks)
            except:
                continue
        raise RuntimeError(f"{type(encoder.tokenizer)} tokenizer cannot decode or detokenize")

    merged_index = 0
    merged_dataset = IndexedDataset(os.path.join(path_to_data, "merge"))

    # sorted to ensure ordering matches merged dataset
    basenames = sorted(
        [
            name
            for name in os.listdir(path_to_data)
            if name.endswith(".idx") and not name.startswith("merge")
        ]
    )

    # index into the merged document index
    merged_doc_index_index = 0

    for basename in basenames:
        realpath_raw = f"{os.path.join(path_to_raws, '_'.join(basename.split('_')[:-2]))}.jsonl"
        realpath_doc = os.path.join(path_to_data, basename.split(".")[-2])

        dataset_index = 0
        dataset = IndexedDataset(realpath_doc)

        merged_doc_idx = merged_dataset.document_indices[
            merged_doc_index_index : merged_doc_index_index + len(dataset.document_indices)
        ]
        merged_doc_idx = merged_doc_idx - merged_doc_idx[0]

        assert (
            dataset.document_indices == merged_doc_idx
        ).all(), f"ERROR: {basename.split('_')[:-2]}: merged dataset document indices mismatch"

        merged_doc_index_index += len(dataset.document_indices) - 1

        with open(realpath_raw, "rt") as reader:
            for json_line in reader:
                toks = encoder.encode(json_line)[0]["text"]

                raw = tokens_to_string(toks)

                processed_toks = []
                while len(processed_toks) < len(toks):
                    processed_toks.extend(dataset[dataset_index])
                    dataset_index += 1
                processed = tokens_to_string(processed_toks)

                assert (
                    raw == processed
                ), f"ERROR: {basename.split('_')[:-2]}: raw and processed documents do not match"

                merged_toks = []
                while len(merged_toks) < len(toks):
                    merged_toks.extend(merged_dataset[merged_index])
                    merged_index += 1
                merged = tokens_to_string(merged_toks)

                assert (
                    raw == merged
                ), f"ERROR: {basename.split('_')[:-2]}: raw and merged documents do not match"

        print(
            f"INFO: {''.join(basename.split('_')[:-2])}: raw, processed, and merged documents match!"
        )

    print("INFO: Success!")


def gpt2_vocab(odir):
    if os.path.exists(__LOCAL_GPT2_VOCAB):
        return __LOCAL_GPT2_VOCAB
    path = os.path.join(odir, "vocab.json")
    with open(path, "wb") as writer:
        writer.write(requests.get(MEGATRON_CONFIG_MAP['GPT2BPETokenizer']['vocab']).content)
    return path


def gpt2_merge(odir):
    if os.path.exists(__LOCAL_GPT2_MERGE):
        return __LOCAL_GPT2_MERGE
    path = os.path.join(odir, "merge.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(MEGATRON_CONFIG_MAP['GPT2BPETokenizer']['merges_file']).content)
    return path


def test_preprocess_data_gpt():
    with tempfile.TemporaryDirectory() as temp_dir:

        # gpt specific args
        gpt_args = [
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--merge-file",
            "/opt/data/tokenizers/megatron/gpt2-merges.txt",
            "--append-eod",
            "--workers",
            "10",
            "--log-interval",
            "1",
        ]

        do_test_preprocess_data(temp_dir, extra_args=gpt_args)


def test_preprocess_data_gpt_optimal_workers():
    with tempfile.TemporaryDirectory() as temp_dir:

        # gpt specific args
        gpt_args = [
            "--input",
            "/opt/data/datasets/dclm/dclm.jsonl",
            "--output-prefix",
            f"{temp_dir}/optimal_workers",
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--merge-file",
            "/opt/data/tokenizers/megatron/gpt2-merges.txt",
            "--append-eod",
            "--workers",
            "2",
            "--log-interval",
            "1",
            "--find-optimal-num-workers",
            "--workers-to-check",
            "2",
            "4",
            "8",
            "--max-documents",
            "1002",
        ]
        sys.argv = ["/opt/megatron-lm/tools/preprocess_data.py"] + gpt_args
        runpy.run_path("/opt/megatron-lm/tools/preprocess_data.py", run_name="__main__")


def bert_vocab(odir):
    if os.path.exists(__LOCAL_BERT_VOCAB):
        return __LOCAL_BERT_VOCAB
    path = os.path.join(odir, "vocab.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB).content)
    return path


@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_preprocess_data_bert():
    with tempfile.TemporaryDirectory() as temp_dir:

        # bert specific args
        bert_args = [
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--split-sentences",
            "--workers",
            "10",
            "--log-interval",
            "1",
            "--partitions",
            "2",
            "--keep-sequential-samples",
        ]

        do_test_preprocess_data(temp_dir, extra_args=bert_args)


if __name__ == "__main__":
    test_preprocess_data_gpt()
    test_preprocess_data_bert()
    test_preprocess_data_gpt_optimal_workers()
