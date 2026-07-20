#!/usr/bin/env python3

"""Print the total token count in a Megatron indexed dataset."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from megatron.core.datasets.indexed_dataset import IndexedDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path",
        help="Indexed dataset prefix, or its .bin or .idx path.",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if data_path.endswith((".bin", ".idx")):
        data_path = data_path[:-4]

    bin_path = Path(f"{data_path}.bin")
    idx_path = Path(f"{data_path}.idx")
    missing = [str(path) for path in (bin_path, idx_path) if not path.is_file()]
    if missing:
        parser.error("missing indexed dataset file(s): " + ", ".join(missing))

    dataset = IndexedDataset(data_path, multimodal=False)
    print(int(np.sum(dataset.sequence_lengths, dtype=np.int64)))


if __name__ == "__main__":
    main()
