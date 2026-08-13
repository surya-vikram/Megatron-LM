#!/usr/bin/env python3

"""Print the number of deterministic packed samples in one metadata epoch."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from megatron.training.datasets.chat_packing import build_pack_index, load_pack_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--mode", required=True, choices=("sft", "simpo"))
    parser.add_argument("--sequence-length", required=True, type=int)
    args = parser.parse_args()

    lengths, metadata = load_pack_lengths(
        args.metadata, expected_mode=args.mode, expected_rows=int(_metadata_rows(args.metadata))
    )
    pack_index = build_pack_index(np.arange(len(lengths)), lengths, args.sequence_length)
    print(len(pack_index))


def _metadata_rows(metadata_path: str) -> int:
    import json
    import os

    with open(os.path.join(metadata_path, "metadata.json"), "r", encoding="utf-8") as reader:
        return int(json.load(reader)["rows"])


if __name__ == "__main__":
    main()
