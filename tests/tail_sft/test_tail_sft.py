"""CPU checks for selection math; runnable without importing CUDA-only MCore.

Run: python -m pytest -q tests/tail_sft
These complement, rather than replace, checkpoint-based GPU integration tests.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tail_sft_math", ROOT / "megatron/training/tail_sft.py"
)
tail = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tail)


def test_disabled_feature_requires_no_new_settings():
    args = SimpleNamespace()
    assert not tail.enabled(args)
    tail.validate_args(args)
    args = SimpleNamespace(
        tail_sft=True, tail_sft_filter_fraction=0.0, tail_sft_filter_schedule="ramp"
    )
    assert not tail.enabled(args)
    tail.validate_args(args)


def test_reference_provenance_is_checked(tmp_path, monkeypatch):
    # Replace only the tokenizer fingerprint dependency to avoid importing MCore.
    dependency = ModuleType("megatron.training.datasets.chat_packing")
    dependency.fingerprint_tokenizer_path = lambda path: "tokenizer-fingerprint"
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    data = tmp_path / "data.jsonl"
    data.write_text('{"messages": []}\n')
    reference = tmp_path / "reference"
    reference.mkdir()
    np.save(reference / "initial_losses.npy", np.array([2.0], dtype=np.float32))
    np.save(reference / "target_counts.npy", np.array([3], dtype=np.int64))
    metadata = dict(
        version=1,
        rows=1,
        dataset_sha256=tail.file_digest(str(data)),
        tokenizer_fingerprint="tokenizer-fingerprint",
        prompt_format="chimera",
        target_mask="sft_tokenizer_all_assistant_turns",
        reference_model="initial-hf",
    )
    (reference / "metadata.json").write_text(json.dumps(metadata))
    scores = tail.ReferenceLosses(str(reference), str(data), "tokenizer", "chimera")
    assert scores.losses.tolist() == [2.0]
    data.write_text('{"messages": ["changed"]}\n')
    with pytest.raises(ValueError, match="dataset_sha256"):
        tail.ReferenceLosses(str(reference), str(data), "tokenizer", "chimera")


@pytest.mark.parametrize(
    "schedule,iteration,expected",
    [("static", 0, 0.5), ("ramp", 0, 0), ("ramp", 5, 0.25), ("ramp", 10, 0.5), ("ramp", 20, 0.5)],
)
def test_schedule(schedule, iteration, expected):
    assert tail.filter_fraction(0.5, schedule, iteration, 11) == expected


def test_margin_selection_and_padding_gradients():
    # Conversation means 2, 4, 6; reference means 5, 4, 5.
    # Drop margin -3, rather than the largest absolute loss.
    losses = torch.tensor([1.0, 3.0, 4.0, 5.0, 7.0, 99.0], requires_grad=True)
    objective, units, report = tail.filtered_loss(
        losses,
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        torch.tensor([0, 2, 3, 5, 6]),
        torch.tensor([5.0, 4.0, 5.0, 0.0]),
        1 / 3,
    )
    assert objective.item() == pytest.approx(16 / 3)
    assert units.item() == 1
    assert report["tail sft dropped fraction"].tolist() == [1.0, 3.0]
    objective.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([0.0, 0.0, 1 / 3, 1 / 3, 1 / 3, 0.0]))


@pytest.mark.parametrize(
    "fraction,expected", [(0.0, [1 / 3, 1 / 3, 1 / 3]), (1.0, [0.0, 0.0, 1.0])]
)
def test_cap_and_stable_ties(fraction, expected):
    losses = torch.ones(3, requires_grad=True)
    objective, _, _ = tail.filtered_loss(
        losses, torch.ones(3), torch.arange(4), torch.ones(3), fraction
    )
    objective.backward()
    torch.testing.assert_close(losses.grad, torch.tensor(expected))


def test_packing_does_not_change_selection_or_gradients():
    packed = torch.tensor([2.0, 4.0, 9.0, 6.0, 3.0, 5.0, 9.0], requires_grad=True)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    objective, _, _ = tail.filtered_loss(
        packed, mask, torch.tensor([0, 2, 3, 4, 6, 7]), torch.tensor([6.0, 0.0, 5.0, 4.0, 0.0]), 0.5
    )
    unpacked = torch.tensor([[2.0, 4.0, 9.0], [6.0, 9.0, 9.0], [3.0, 5.0, 9.0]], requires_grad=True)
    other, _, _ = tail.filtered_loss(
        unpacked,
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        torch.tensor([0, 3, 6, 9]),
        torch.tensor([6.0, 5.0, 4.0]),
        0.5,
    )
    torch.testing.assert_close(objective, other)
    objective.backward()
    other.backward()
    torch.testing.assert_close(
        packed.grad[mask.bool()],
        unpacked.grad[
            torch.tensor([[True, True, False], [True, False, False], [True, True, False]])
        ],
    )


def test_accumulation_averages_microbatch_means():
    parameter = torch.tensor(2.0, requires_grad=True)
    for factors in (torch.tensor([1.0, 3.0]), torch.tensor([5.0, 7.0, 9.0])):
        objective, _, _ = tail.filtered_loss(
            parameter * factors,
            torch.ones_like(factors),
            torch.arange(len(factors) + 1),
            torch.zeros_like(factors),
            0,
        )
        objective.backward()
    # Existing Megatron finalizer divides by the two normalization units.
    assert (parameter.grad / 2).item() == pytest.approx((2 + 7) / 2)


@pytest.mark.parametrize("boundaries", [[1, 3], [0, 2], [0, 2, 2, 3]])
def test_invalid_boundaries_fail(boundaries):
    with pytest.raises(ValueError, match="boundar"):
        tail.sequence_losses(torch.ones(3), torch.ones(3), torch.tensor(boundaries))


def _distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Variable conversation counts; rank zero loses every local conversation.
        factors = torch.tensor([1.0]) if rank == 0 else torch.tensor([3.0, 5.0])
        parameter = torch.tensor(2.0, requires_grad=True)
        objective, _, report = tail.filtered_loss(
            parameter * factors,
            torch.ones_like(factors),
            torch.arange(len(factors) + 1),
            torch.zeros_like(factors),
            2 / 3,
        )
        objective.backward()
        assert report["tail sft dropped fraction"][0].item() == 1
        # Sum gradients, then divide by DP * microbatch units, as Megatron does.
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(parameter.grad / 2, torch.tensor(5.0))
    finally:
        dist.destroy_process_group()


def test_distributed_variable_counts_and_empty_local_survivors(tmp_path):
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def test_segment_reduction_preserves_small_conversation_losses():
    losses = torch.tensor([1e8, 1.0, 1.0], requires_grad=True)
    sums, counts = tail.sequence_losses(losses, torch.ones(3), torch.tensor([0, 1, 3]))
    assert sums.tolist() == [1e8, 2.0]
    assert counts.tolist() == [1.0, 2.0]
    objective, _, _ = tail.filtered_loss(
        losses, torch.ones(3), torch.tensor([0, 1, 3]), torch.tensor([1e8, 1.0]), 0.5
    )
    objective.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([0.0, 0.5, 0.5]))


def test_packed_yarn_fusion_override_is_opt_in():
    args = SimpleNamespace(
        tail_sft=True,
        tail_sft_filter_fraction=0.5,
        tail_sft_filter_schedule="ramp",
        sft=True,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        train_iters=5,
        tail_sft_reference_losses="reference",
        num_experts=32,
        pack_samples=True,
        position_embedding_type="yarn",
        apply_rope_fusion=True,
    )
    tail.validate_args(args)
    assert args.apply_rope_fusion is False
    args.tail_sft_filter_fraction = 0
    args.apply_rope_fusion = True
    tail.validate_args(args)
    assert args.apply_rope_fusion is True
