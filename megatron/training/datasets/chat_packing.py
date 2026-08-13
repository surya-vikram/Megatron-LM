# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared deterministic packing helpers for chat training datasets."""

from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from typing import Iterable, Sequence

import numpy as np


INVALID_SAMPLE_LENGTH = -1
PACK_METADATA_VERSION = 1
TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


@dataclass(frozen=True)
class PackIndex:
    """Immutable mapping from packed samples to source-row indices."""

    offsets: np.ndarray
    row_indices: np.ndarray
    oversized_row_indices: np.ndarray
    invalid_row_count: int
    packed_token_count: int

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def rows_for_pack(self, pack_index: int) -> np.ndarray:
        start = int(self.offsets[pack_index])
        end = int(self.offsets[pack_index + 1])
        return self.row_indices[start:end]


def build_pack_index(
    source_row_indices: Iterable[int], sample_lengths: Sequence[int], capacity: int
) -> PackIndex:
    """Pack rows greedily without dropping a row that does not fit the current pack."""

    if capacity <= 0:
        raise ValueError(f"packing capacity must be positive, got {capacity}")

    offsets = [0]
    packed_rows = []
    oversized_rows = []
    invalid_row_count = 0
    packed_token_count = 0
    current_length = 0

    for raw_row_index in source_row_indices:
        row_index = int(raw_row_index)
        sample_length = int(sample_lengths[row_index])
        if sample_length == INVALID_SAMPLE_LENGTH:
            invalid_row_count += 1
            continue
        if sample_length <= 0:
            raise ValueError(
                f"row {row_index} has invalid packable length {sample_length}; "
                f"use {INVALID_SAMPLE_LENGTH} for rejected rows"
            )
        if sample_length > capacity:
            oversized_rows.append(row_index)
            continue

        if current_length and current_length + sample_length > capacity:
            offsets.append(len(packed_rows))
            current_length = 0

        packed_rows.append(row_index)
        packed_token_count += sample_length
        current_length += sample_length

    if current_length:
        offsets.append(len(packed_rows))

    if len(offsets) == 1:
        raise ValueError(
            "no packable chat samples remain after filtering malformed and oversized rows"
        )

    return PackIndex(
        offsets=np.asarray(offsets, dtype=np.int64),
        row_indices=np.asarray(packed_rows, dtype=np.int64),
        oversized_row_indices=np.asarray(oversized_rows, dtype=np.int64),
        invalid_row_count=invalid_row_count,
        packed_token_count=packed_token_count,
    )


def load_pack_lengths(
    metadata_path: str, *, expected_mode: str, expected_rows: int
) -> tuple[np.ndarray, dict]:
    """Load an mmap-backed length manifest and validate its immutable source metadata."""

    metadata_path = os.path.abspath(metadata_path)
    metadata_file = os.path.join(metadata_path, "metadata.json")
    lengths_file = os.path.join(metadata_path, "lengths.npy")
    offsets_file = os.path.join(metadata_path, "row_offsets.npy")
    if not all(os.path.isfile(path) for path in (metadata_file, lengths_file, offsets_file)):
        raise FileNotFoundError(
            f"chat packing metadata is incomplete at {metadata_path!r}; expected "
            "metadata.json, lengths.npy, and row_offsets.npy"
        )

    with open(metadata_file, "r", encoding="utf-8") as reader:
        metadata = json.load(reader)

    if metadata.get("version") != PACK_METADATA_VERSION:
        raise ValueError(
            f"unsupported chat packing metadata version {metadata.get('version')!r}; "
            f"expected {PACK_METADATA_VERSION}"
        )
    if metadata.get("mode") != expected_mode:
        raise ValueError(
            f"packing metadata mode is {metadata.get('mode')!r}, expected {expected_mode!r}"
        )

    lengths = np.load(lengths_file, mmap_mode="r")
    if lengths.ndim != 1 or len(lengths) != expected_rows:
        raise ValueError(
            f"packing metadata has shape {lengths.shape}, expected ({expected_rows},)"
        )
    return lengths, metadata


class IndexedJsonlDataset:
    """Random-access JSONL reader backed by precomputed byte offsets and mmap."""

    def __init__(self, dataset_path: str, metadata_path: str, field: str | None = None):
        self.dataset_path = os.path.abspath(dataset_path)
        self.row_offsets = np.load(
            os.path.join(os.path.abspath(metadata_path), "row_offsets.npy"), mmap_mode="r"
        )
        self.field = field
        self._file = None
        self._mmap = None

    def __len__(self) -> int:
        return len(self.row_offsets)

    def __getitem__(self, index: int):
        if self._mmap is None:
            self._file = open(self.dataset_path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._mmap.seek(int(self.row_offsets[index]))
        row = json.loads(self._mmap.readline())
        return row if self.field is None else row.get(self.field)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_mmap"] = None
        return state

    def __del__(self):
        mmap_handle = getattr(self, "_mmap", None)
        file_handle = getattr(self, "_file", None)
        if mmap_handle is not None:
            mmap_handle.close()
        if file_handle is not None:
            file_handle.close()


def validate_metadata_source(metadata: dict, dataset_path: str) -> None:
    """Reject a manifest when the source JSONL has changed since preprocessing."""

    stat = os.stat(dataset_path)
    expected_size = metadata.get("source_size")
    expected_mtime_ns = metadata.get("source_mtime_ns")
    if stat.st_size != expected_size or stat.st_mtime_ns != expected_mtime_ns:
        raise ValueError(
            "chat packing metadata does not match the current JSONL size/mtime; "
            "rerun examples/chimera/prepare_chat_data.py"
        )


def fingerprint_tokenizer_path(tokenizer_path: str) -> str:
    """Hash tokenizer artifacts that can change chat tokenization or masking."""

    digest = hashlib.sha256()
    for filename in TOKENIZER_FINGERPRINT_FILES:
        path = os.path.join(tokenizer_path, filename)
        if not os.path.isfile(path):
            continue
        digest.update(filename.encode("utf-8"))
        with open(path, "rb") as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
