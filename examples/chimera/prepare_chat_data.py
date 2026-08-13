#!/usr/bin/env python3

"""Build reusable exact token-length metadata for packed Chimera SFT or SimPO."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer
from megatron.training.datasets.chat_packing import (
    INVALID_SAMPLE_LENGTH,
    PACK_METADATA_VERSION,
    fingerprint_tokenizer_path,
)


IGNORE_INDEX = -100


_tokenizer = None
_mode = None


def _validate_chat_messages(messages):
    if not isinstance(messages, list) or not messages:
        return False
    return all(
        isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant"}
        and isinstance(message.get("content"), str)
        for message in messages
    )


def _initialize_worker(tokenizer_model: str, prompt_format: str, mode: str) -> None:
    global _tokenizer, _mode
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _tokenizer = SFTTokenizer(tokenizer_model, prompt_format, load_model_config=False)
    _mode = mode


def _conversation_length(messages) -> int:
    if not _validate_chat_messages(messages):
        return INVALID_SAMPLE_LENGTH
    tokens, targets = _tokenizer.tokenize_conversation(
        messages, return_target=True, add_generation_prompt=False
    )
    if len(tokens) < 2:
        return INVALID_SAMPLE_LENGTH
    if not np.any((targets != IGNORE_INDEX) & (targets != _tokenizer.pad_id)):
        return INVALID_SAMPLE_LENGTH
    return int(len(tokens) - 1)


def _measure_line(line: str) -> int:
    try:
        row = json.loads(line)
    except (TypeError, json.JSONDecodeError):
        return INVALID_SAMPLE_LENGTH

    if _mode == "sft":
        if not isinstance(row, dict):
            return INVALID_SAMPLE_LENGTH
        return _conversation_length(row.get("messages"))

    if not isinstance(row, dict) or "chosen" not in row or "rejected" not in row:
        return INVALID_SAMPLE_LENGTH
    chosen_length = _conversation_length(row["chosen"])
    rejected_length = _conversation_length(row["rejected"])
    if chosen_length == INVALID_SAMPLE_LENGTH or rejected_length == INVALID_SAMPLE_LENGTH:
        return INVALID_SAMPLE_LENGTH
    return chosen_length + rejected_length


def _scan_nonempty_offsets(path: str) -> np.ndarray:
    offsets = []
    with open(path, "rb") as reader:
        while True:
            offset = reader.tell()
            line = reader.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return np.asarray(offsets, dtype=np.int64)


def _lines_at_offsets(path: str, offsets: np.ndarray):
    with open(path, "rb") as reader:
        for offset in offsets:
            reader.seek(int(offset))
            yield reader.readline().decode("utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input SFT or SimPO JSONL")
    parser.add_argument("--output", required=True, help="Output metadata directory")
    parser.add_argument("--mode", required=True, choices=("sft", "simpo"))
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--prompt-format", default="chimera")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PREPARE_WORKERS", 32)))
    parser.add_argument("--chunksize", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    source_stat = os.stat(input_path)
    tokenizer_fingerprint = fingerprint_tokenizer_path(args.tokenizer_model)
    row_offsets = _scan_nonempty_offsets(input_path)
    total_rows = len(row_offsets)
    if total_rows == 0:
        raise ValueError(f"input JSONL contains no rows: {input_path}")

    print(
        f"Preparing {args.mode} packing metadata: rows={total_rows} workers={args.workers}",
        flush=True,
    )
    started = time.monotonic()
    lengths = np.empty(total_rows, dtype=np.int32)
    context = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with context.Pool(
        processes=args.workers,
        initializer=_initialize_worker,
        initargs=(args.tokenizer_model, args.prompt_format, args.mode),
    ) as pool:
        results = pool.imap(
            _measure_line,
            _lines_at_offsets(input_path, row_offsets),
            chunksize=args.chunksize,
        )
        for index, length in enumerate(results):
            lengths[index] = length
            completed = index + 1
            if completed % 1000 == 0 or completed == total_rows:
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                remaining = (total_rows - completed) / rate if rate else 0.0
                print(
                    f"  rows={completed}/{total_rows} rate={rate:.1f}/s "
                    f"eta={remaining / 60:.1f}m",
                    flush=True,
                )

    final_source_stat = os.stat(input_path)
    if (
        final_source_stat.st_size != source_stat.st_size
        or final_source_stat.st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise RuntimeError("input JSONL changed during packing metadata preparation; rerun it")
    if fingerprint_tokenizer_path(args.tokenizer_model) != tokenizer_fingerprint:
        raise RuntimeError("tokenizer artifacts changed during packing metadata preparation; rerun it")

    os.makedirs(output_path, exist_ok=True)
    lengths_tmp = os.path.join(output_path, "lengths.npy.tmp")
    lengths_path = os.path.join(output_path, "lengths.npy")
    with open(lengths_tmp, "wb") as writer:
        np.save(writer, lengths, allow_pickle=False)
    os.replace(lengths_tmp, lengths_path)

    offsets_tmp = os.path.join(output_path, "row_offsets.npy.tmp")
    offsets_path = os.path.join(output_path, "row_offsets.npy")
    with open(offsets_tmp, "wb") as writer:
        np.save(writer, row_offsets, allow_pickle=False)
    os.replace(offsets_tmp, offsets_path)

    metadata = {
        "version": PACK_METADATA_VERSION,
        "mode": args.mode,
        "source_path": input_path,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "rows": total_rows,
        "valid_rows": int(np.count_nonzero(lengths != INVALID_SAMPLE_LENGTH)),
        "invalid_rows": int(np.count_nonzero(lengths == INVALID_SAMPLE_LENGTH)),
        "tokenizer_model": os.path.abspath(args.tokenizer_model),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "prompt_format": args.prompt_format,
    }
    metadata_tmp = os.path.join(output_path, "metadata.json.tmp")
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_tmp, "w", encoding="utf-8") as writer:
        json.dump(metadata, writer, indent=2, sort_keys=True)
        writer.write("\n")
    os.replace(metadata_tmp, metadata_path)

    elapsed = time.monotonic() - started
    print(
        f"Wrote {output_path}: valid={metadata['valid_rows']} "
        f"invalid={metadata['invalid_rows']} elapsed={elapsed / 60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
