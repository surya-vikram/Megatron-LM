"""Exercise actual SFT dataset methods without initializing CUDA-only MCore."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch


@pytest.fixture
def dataset_classes():
    path = Path(__file__).resolve().parents[2] / "megatron/training/datasets/sft_dataset.py"
    tree = ast.parse(path.read_text())
    # Execute the production classes/functions. Only dependency imports are
    # replaced; tokenization, packing, masking and collation methods are intact.
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(
        np=np,
        torch=torch,
        os=os,
        Any=Any,
        Dict=Dict,
        Optional=Optional,
        MegatronDataset=object,
        LowLevelDataset=object,
        GPTDatasetConfig=object,
        Split=object,
        get_args=lambda: SimpleNamespace(pack_samples=False),
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["SFTDataset"], namespace["PackSamplesCollator"]


class Tokenizer:
    pad = 0
    eod = 9

    def tokenize_conversation(self, messages, **kwargs):
        marker = int(messages[0]["content"])
        return np.array([10, 11, marker, 9]), np.array([-100, -100, marker, 9])


def make_dataset(cls, packing, reference=True):
    dataset = object.__new__(cls)
    dataset.dataset = [[{"role": "assistant", "content": str(marker)}] for marker in (20, 40, 60)]
    dataset.indices = np.array([2, 0, 1])
    dataset.config = SimpleNamespace(
        tokenizer=Tokenizer(),
        sequence_length=8,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
    )
    dataset._pack_samples = packing
    # Special methods are resolved on the type, rather than on an instance.
    dataset._pack_index = type(
        "Index",
        (),
        {
            "__len__": lambda self: 2,
            "rows_for_pack": lambda self, index: [2, 0] if index == 0 else [1],
        },
    )()
    dataset._pack_lengths = np.array([3, 3, 3])
    dataset._stats = dict.fromkeys(
        [
            "steps",
            "total_packed",
            "total_active_tok",
            "total_pad_tok",
            "total_tok",
            "skipped_oversized",
            "skipped_malformed",
        ],
        0,
    )
    if reference:
        dataset._tail_reference = SimpleNamespace(
            losses=np.array([1.0, 2.0, 3.0]), lengths=np.array([2, 2, 2])
        )
    return dataset


def test_packed_metadata_alignment_and_ordinary_outputs(dataset_classes):
    cls, collator = dataset_classes
    tail = make_dataset(cls, True)
    ordinary = make_dataset(cls, True, reference=False)
    for index in range(2):
        filtered = tail[index]
        baseline = ordinary[index]
        assert "tail_sft_initial_losses" not in baseline
        for key in baseline:
            torch.testing.assert_close(filtered[key], baseline[key])
    batch = collator()([tail[0], tail[1]])
    assert batch["cu_seqlens"].tolist() == [[0, 3, 6, 8, 11, 16]]
    assert batch["tail_sft_initial_losses"].tolist() == [3.0, 1.0, 0.0, 2.0, 0.0]
    assert batch["loss_mask"].sum().item() == 6


def test_unpacked_physical_row_and_mask_match_ordinary(dataset_classes):
    cls, _ = dataset_classes
    tail = make_dataset(cls, False)
    ordinary = make_dataset(cls, False, reference=False)
    for index, expected in enumerate([3.0, 1.0, 2.0]):
        filtered = tail[index]
        baseline = ordinary[index]
        for key in baseline:
            torch.testing.assert_close(filtered[key], baseline[key])
        assert filtered["tail_sft_initial_losses"].item() == expected


def test_mask_mismatch_fails(dataset_classes):
    cls, _ = dataset_classes
    dataset = make_dataset(cls, False)
    dataset._tail_reference.lengths[2] = 1
    with pytest.raises(ValueError, match="target mask changed"):
        dataset[0]


def test_oversized_rows_are_not_silently_replaced(dataset_classes):
    cls, _ = dataset_classes
    dataset = make_dataset(cls, False)
    dataset.config.sequence_length = 2
    with pytest.raises(ValueError, match="exceeds sequence length"):
        dataset[0]
