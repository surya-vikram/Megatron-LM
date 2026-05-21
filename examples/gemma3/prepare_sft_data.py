"""Prepare bundled SFT datasets for Gemma3 smoke, overfit, and held-out evals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from examples.gemma3.sft_data_assets import (
    GOLD_MEDICAL_SAMPLES,
    REASONING_EVAL_TASKS,
    build_pair_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Gemma3 SFT data bundles.")
    parser.add_argument(
        "--source",
        choices=["capybara", "jsonl"],
        default="capybara",
        help="Dataset source. 'jsonl' expects preformatted or convertible records.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
        help="Input JSONL when --source jsonl.",
    )
    parser.add_argument(
        "--capybara-split",
        type=str,
        default="train[:100]",
        help="Subset expression used when --source capybara.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/gemma3/data_bundle",
        help="Directory where bundle files will be written.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Tokenizer used for chat templating and filtering.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=16384,
        help="Maximum tokenized chat length kept in the bundle.",
    )
    parser.add_argument(
        "--heldout-count",
        type=int,
        default=8,
        help="Number of filtered source samples reserved for held-out eval.",
    )
    parser.add_argument(
        "--smoke-train-count",
        type=int,
        default=32,
        help="Number of training samples used for smoke runs.",
    )
    parser.add_argument(
        "--overfit-pack-size",
        type=int,
        default=4,
        help="Number of gold samples used in the tiny-pack overfit run.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap on train.jsonl sample count after splitting. 0 keeps all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Deterministic shuffle seed before split.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle filtered source records before splitting.",
    )
    parser.add_argument(
        "--inject-gold-into-train",
        action="store_true",
        help="Append unused gold overfit samples to train.jsonl.",
    )
    return parser.parse_args()


def normalize_record(raw_record: dict) -> dict | None:
    if "messages" in raw_record:
        messages = raw_record["messages"]
    elif "conversation" in raw_record:
        messages = []
        for turn in raw_record["conversation"]:
            if turn.get("input"):
                messages.append({"role": "user", "content": turn["input"]})
            if turn.get("output"):
                messages.append({"role": "model", "content": turn["output"]})
    elif "user" in raw_record and "model" in raw_record:
        messages = [
            {"role": "user", "content": raw_record["user"]},
            {"role": "model", "content": raw_record["model"]},
        ]
    else:
        return None

    if not messages:
        return None

    return {"messages": messages}


def load_source_records(args: argparse.Namespace) -> list[dict]:
    if args.source == "capybara":
        dataset = load_dataset("LDJnr/Capybara", split=args.capybara_split)
        return [normalize_record(record) for record in dataset]

    if not args.input_path:
        raise ValueError("--input-path is required when --source jsonl")

    records = []
    with open(args.input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(normalize_record(json.loads(line)))
    return records


def filter_records(records: Iterable[dict | None], tokenizer, max_seq_length: int) -> tuple[list[dict], int]:
    filtered = []
    skipped = 0
    for record in records:
        if record is None:
            continue
        rendered = tokenizer.apply_chat_template(record["messages"], tokenize=False)
        tokenized = tokenizer(rendered).input_ids
        if len(tokenized) <= max_seq_length:
            filtered.append(record)
        else:
            skipped += 1
    return filtered, skipped


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    source_records = load_source_records(args)
    filtered_records, skipped_long = filter_records(source_records, tokenizer, args.max_seq_length)
    if not filtered_records:
        raise RuntimeError("No usable records remained after filtering.")

    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(filtered_records)

    if len(filtered_records) <= args.heldout_count:
        raise RuntimeError(
            f"Need more than {args.heldout_count} filtered records, found {len(filtered_records)}."
        )

    heldout_records = filtered_records[-args.heldout_count :]
    train_records = filtered_records[: -args.heldout_count]
    if args.max_train_samples > 0:
        train_records = train_records[: args.max_train_samples]

    smoke_train_records = train_records[: min(args.smoke_train_count, len(train_records))]
    if not smoke_train_records:
        raise RuntimeError("Smoke train split is empty.")

    overfit_single_records = [build_pair_sample(GOLD_MEDICAL_SAMPLES[0]["user"], GOLD_MEDICAL_SAMPLES[0]["model"])]
    overfit_pack_records = [
        build_pair_sample(sample["user"], sample["model"])
        for sample in GOLD_MEDICAL_SAMPLES[: args.overfit_pack_size]
    ]

    if args.inject_gold_into_train:
        train_records = train_records + overfit_pack_records
        smoke_train_records = smoke_train_records + overfit_pack_records

    train_path = output_dir / "train.jsonl"
    smoke_train_path = output_dir / "smoke_train.jsonl"
    heldout_path = output_dir / "heldout.jsonl"
    overfit_single_path = output_dir / "overfit_single.jsonl"
    overfit_pack_path = output_dir / "overfit_pack.jsonl"
    reasoning_eval_path = output_dir / "reasoning_eval.json"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(train_path, train_records)
    write_jsonl(smoke_train_path, smoke_train_records)
    write_jsonl(heldout_path, heldout_records)
    write_jsonl(overfit_single_path, overfit_single_records)
    write_jsonl(overfit_pack_path, overfit_pack_records)
    write_json(reasoning_eval_path, REASONING_EVAL_TASKS)

    manifest = {
        "source": args.source,
        "input_path": args.input_path,
        "tokenizer_model": args.tokenizer_model,
        "max_seq_length": args.max_seq_length,
        "skipped_long_samples": skipped_long,
        "counts": {
            "filtered_source": len(filtered_records),
            "train": len(train_records),
            "smoke_train": len(smoke_train_records),
            "heldout": len(heldout_records),
            "overfit_single": len(overfit_single_records),
            "overfit_pack": len(overfit_pack_records),
            "reasoning_eval": len(REASONING_EVAL_TASKS),
        },
        "paths": {
            "train": str(train_path),
            "smoke_train": str(smoke_train_path),
            "heldout": str(heldout_path),
            "overfit_single": str(overfit_single_path),
            "overfit_pack": str(overfit_pack_path),
            "reasoning_eval": str(reasoning_eval_path),
        },
    }
    write_json(manifest_path, manifest)

    print("--- Gemma3 SFT Data Bundle ---")
    print(f"Output Dir: {output_dir}")
    print(f"Filtered Source Samples: {len(filtered_records)}")
    print(f"Skipped Long Samples: {skipped_long}")
    print(f"Train Samples: {len(train_records)}")
    print(f"Smoke Samples: {len(smoke_train_records)}")
    print(f"Held-out Samples: {len(heldout_records)}")
    print(f"Overfit Single Samples: {len(overfit_single_records)}")
    print(f"Overfit Pack Samples: {len(overfit_pack_records)}")
    print(f"Reasoning Eval Tasks: {len(REASONING_EVAL_TASKS)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
