# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest

from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer
from megatron.training.datasets import sft_dataset as sft_module
from megatron.training.datasets import simpo_dataset as simpo_module
from megatron.training.datasets.chat_packing import build_pack_index
from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset
from megatron.training.datasets.simpo_dataset import PackSamplesCollator, SimPODataset


class FakeTokenizer:
    pad = 0
    eod = 9

    def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
        del return_target, add_generation_prompt
        content = " ".join(message["content"] for message in conversation)
        if "oversized" in content:
            tokens = np.arange(20, dtype=np.int64) + 10
        else:
            marker = 40 if "second" in content else 20
            tokens = np.array([10, 11, marker, marker + 1], dtype=np.int64)
        targets = tokens.copy()
        targets[:2] = IGNORE_INDEX
        return tokens, targets


def dataset_args(**overrides):
    values = {
        "pack_samples": False,
        "pack_metadata_path": None,
        "simpo": False,
        "debug_dataset": False,
        "log_dataset_stats": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_dataset(dataset_type, rows, num_samples=6, sequence_length=8):
    dataset = object.__new__(dataset_type)
    dataset.dataset = rows
    dataset.dataset_path = "fixture.jsonl"
    dataset.indices = np.arange(len(rows), dtype=np.int32)
    dataset.num_samples = num_samples
    dataset.config = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        sequence_length=sequence_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        create_attention_mask=False,
    )
    dataset._stats = {
        "steps": 0,
        "total_packed": 0,
        "total_active_tok": 0,
        "total_pad_tok": 0,
        "total_tok": 0,
        "skipped_oversized": 0,
        "skipped_malformed": 0,
    }
    dataset._pack_samples = False
    dataset._pack_index = None
    if dataset_type is SimPODataset:
        dataset.tokenizer = dataset.config.tokenizer
    return dataset


def conversation(content):
    return [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": content},
    ]


def preference_pair(content):
    return {
        "chosen": conversation(content),
        "rejected": conversation(f"rejected {content}"),
    }


def test_sft_repeats_physical_rows_to_requested_length(monkeypatch):
    monkeypatch.setattr(sft_module, "get_args", lambda: dataset_args())
    dataset = make_dataset(
        SFTDataset,
        [conversation("first"), conversation("second")],
        num_samples=6,
    )

    assert len(dataset) == 6
    assert dataset[0]["tokens"][2].item() == 20
    assert dataset[1]["tokens"][2].item() == 40
    assert dataset[2]["tokens"][2].item() == 20
    dataset.num_samples = None
    assert len(dataset) == 2


def test_sft_skips_malformed_row_and_fails_when_none_are_usable(monkeypatch):
    monkeypatch.setattr(sft_module, "get_args", lambda: dataset_args())
    dataset = make_dataset(SFTDataset, [None, conversation("first")])
    assert dataset[0]["loss_mask"].sum().item() > 0

    dataset = make_dataset(SFTDataset, [None, []])
    with pytest.raises(RuntimeError, match="malformed=2, oversized=0"):
        dataset[0]


def test_sft_all_oversized_fails_after_one_traversal(monkeypatch):
    monkeypatch.setattr(sft_module, "get_args", lambda: dataset_args())
    dataset = make_dataset(
        SFTDataset,
        [conversation("oversized one"), conversation("oversized two")],
    )

    with pytest.raises(RuntimeError, match="malformed=0, oversized=2"):
        dataset[0]


def test_simpo_repeats_physical_rows_to_requested_length(monkeypatch):
    monkeypatch.setattr(simpo_module, "get_args", lambda: dataset_args(simpo=True))
    dataset = make_dataset(
        SimPODataset,
        [preference_pair("first"), preference_pair("second")],
        num_samples=6,
    )

    assert len(dataset) == 6
    assert dataset[0]["tokens"][2].item() == 20
    assert dataset[1]["tokens"][2].item() == 40
    assert dataset[2]["tokens"][2].item() == 20
    dataset.num_samples = None
    assert len(dataset) == 2


def test_simpo_skips_malformed_and_oversized_rows_without_looping(monkeypatch):
    monkeypatch.setattr(simpo_module, "get_args", lambda: dataset_args(simpo=True))
    malformed = {"chosen": conversation("first")}
    oversized = preference_pair("oversized")
    dataset = make_dataset(SimPODataset, [malformed, oversized])

    with pytest.raises(RuntimeError, match="malformed=1, oversized=1"):
        dataset[0]


def test_pack_index_carries_non_fitting_sample_into_next_pack():
    pack_index = build_pack_index(np.arange(3), np.array([6, 6, 4]), capacity=10)

    assert pack_index.rows_for_pack(0).tolist() == [0]
    assert pack_index.rows_for_pack(1).tolist() == [1, 2]
    assert np.concatenate(
        [pack_index.rows_for_pack(i) for i in range(len(pack_index))]
    ).tolist() == [0, 1, 2]


def test_pack_index_only_skips_samples_larger_than_empty_pack():
    pack_index = build_pack_index(np.arange(4), np.array([4, -1, 11, 6]), capacity=10)

    assert pack_index.row_indices.tolist() == [0, 3]
    assert pack_index.oversized_row_indices.tolist() == [2]
    assert pack_index.invalid_row_count == 1
    assert pack_index.packed_token_count == 10


def test_packed_sft_resets_positions_and_shifts_each_conversation_independently():
    dataset = make_dataset(
        SFTDataset,
        [conversation("first"), conversation("second"), conversation("first again")],
        num_samples=None,
        sequence_length=8,
    )
    dataset._pack_samples = True
    dataset._pack_lengths = np.array([3, 3, 3], dtype=np.int32)
    dataset._pack_index = build_pack_index(
        dataset.indices, dataset._pack_lengths, dataset.config.sequence_length
    )

    item = dataset[0]

    assert item["cu_seqlens"].tolist() == [0, 3, 6, 8]
    assert item["position_ids"].tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert item["labels"].tolist()[:6] == [IGNORE_INDEX, 20, 21, IGNORE_INDEX, 40, 41]
    assert item["loss_mask"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_packed_simpo_keeps_pairs_adjacent_across_microbatch_collation():
    rows = [preference_pair("first"), preference_pair("second")] * 2
    dataset = make_dataset(SimPODataset, rows, num_samples=None, sequence_length=16)
    dataset._pack_samples = True
    dataset._pack_lengths = np.full(4, 6, dtype=np.int32)
    dataset._pack_index = build_pack_index(
        dataset.indices, dataset._pack_lengths, dataset.config.sequence_length
    )

    first = dataset[0]
    second = dataset[1]
    batch = PackSamplesCollator()([first, second])

    # Eight real sequences (four chosen/rejected pairs) precede both padding segments.
    assert batch["cu_seqlens"].tolist() == [
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 28, 32]
    ]
    assert batch["position_ids"].reshape(-1).tolist()[:12] == [
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
    ]
    for sequence_index in range(8):
        start = batch["cu_seqlens"][0, sequence_index].item()
        end = batch["cu_seqlens"][0, sequence_index + 1].item()
        assert batch["loss_mask"].reshape(-1)[start:end].sum().item() == 2


def test_chimera_masking_keeps_every_assistant_turn_active():
    class FakeHFTokenizer:
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return np.array([[10, 100, 21, 22, 200, 11, 100, 31, 200]])

    tokenizer = object.__new__(SFTTokenizer)
    tokenizer._tokenizer = FakeHFTokenizer()
    tokenizer._prompt_format = "chimera"
    tokenizer._assistant_header = [100]
    tokenizer._prompt_config = SimpleNamespace(
        has_system_role=True,
        custom_chat_template="unused",
        terminator_id=200,
    )

    _, targets = tokenizer.tokenize_conversation(
        conversation("two turns"), return_target=True, add_generation_prompt=False
    )

    assert targets.tolist() == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        21,
        22,
        200,
        IGNORE_INDEX,
        IGNORE_INDEX,
        31,
        200,
    ]


def test_zero_token_gpt_dataset_fails_instead_of_looping():
    dataset = object.__new__(GPTDataset)
    dataset.index_split = SimpleNamespace(name="train")
    dataset.dataset_path = "empty_text_document"
    dataset.num_samples = 10

    with pytest.raises(ValueError, match="contains zero tokens"):
        dataset._get_num_epochs(0)
