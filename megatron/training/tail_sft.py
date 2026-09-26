# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sequence selection for TailSFT, independent of the model and packing layout."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def enabled(args) -> bool:
    """Return whether the optional filtering path is active (zero is ordinary SFT)."""
    return getattr(args, "tail_sft", False) and args.tail_sft_filter_fraction > 0


def file_digest(path: str) -> str:
    """Hash file contents without loading the whole file into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_fraction(target: float, schedule: str, iteration: int, train_iters: int) -> float:
    """Compute the drop fraction using the number of completed optimizer steps."""
    if not 0 <= target <= 1 or schedule not in ("static", "ramp"):
        raise ValueError("TailSFT requires a fraction in [0, 1] and a static/ramp schedule")
    if schedule == "static":
        return target
    return target * min(max(iteration, 0) / max(train_iters - 1, 1), 1.0)


def validate_args(args) -> None:
    """Validate only the active TailSFT path; leave ordinary SFT untouched."""
    if not getattr(args, "tail_sft", False):
        return
    filter_fraction(args.tail_sft_filter_fraction, args.tail_sft_filter_schedule, 0, 1)
    if not enabled(args):
        return
    if (
        getattr(args, "pack_samples", False)
        and getattr(args, "position_embedding_type", None) == "yarn"
    ):
        # The existing fused THD RoPE kernel cannot apply YaRN's mscale.
        # Keep this correction scoped to active TailSFT.
        args.apply_rope_fusion = False
    requirements = (
        (args.sft and not getattr(args, "simpo", False), "TailSFT requires --sft"),
        (args.context_parallel_size == 1, "TailSFT currently requires context parallel size 1"),
        (args.calculate_per_token_loss, "TailSFT requires --calculate-per-token-loss"),
        (bool(args.train_iters), "TailSFT requires explicit --train-iters for its schedule"),
        (bool(args.tail_sft_reference_losses), "TailSFT requires --tail-sft-reference-losses"),
        (not getattr(args, "modelopt_enabled", False), "TailSFT does not support ModelOpt losses"),
        (
            not getattr(args, "mtp_num_layers", None),
            "TailSFT does not support MTP auxiliary losses",
        ),
        (
            not getattr(args, "overlap_moe_expert_parallel_comm", False),
            "TailSFT does not support schedule-plan execution",
        ),
        (
            getattr(args, "cuda_graph_impl", "none") != "full_iteration",
            "TailSFT does not support full-iteration CUDA graphs",
        ),
    )
    for valid, message in requirements:
        if not valid:
            raise ValueError(message)
    if getattr(args, "num_experts", None):
        for name in ("moe_aux_loss_coeff", "moe_z_loss_coeff", "moe_router_bias_update_rate"):
            if getattr(args, name, 0):
                raise ValueError(
                    f"TailSFT requires {name}=0 so discarded sequences cannot update the router"
                )


class ReferenceLosses:
    """Load immutable scores indexed by physical JSONL row, checking their provenance."""

    def __init__(
        self,
        directory: str,
        dataset_path: str,
        tokenizer_path: str,
        prompt_format: str,
        *,
        packing: bool | None = None,
        sequence_length: int | None = None,
        micro_batch_size: int | None = None,
        apply_rope_fusion: bool | None = None,
    ) -> None:
        from megatron.training.datasets.chat_packing import fingerprint_tokenizer_path

        root = Path(directory)
        metadata = json.loads((root / "metadata.json").read_text())
        expected = {
            "version": 1,
            "dataset_sha256": file_digest(dataset_path),
            "tokenizer_fingerprint": fingerprint_tokenizer_path(tokenizer_path),
            "prompt_format": prompt_format,
            "target_mask": "sft_tokenizer_all_assistant_turns",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"TailSFT reference metadata mismatch for {key}; rescore the starting checkpoint"
                )
        if metadata.get("backend") == "megatron":
            for key, value in {
                "packing": packing,
                "sequence_length": sequence_length,
                "micro_batch_size": micro_batch_size,
                "apply_rope_fusion": apply_rope_fusion,
            }.items():
                if value is not None and metadata.get(key) != value:
                    raise ValueError(
                        f"TailSFT native reference mismatch for {key}; rescore with the training layout"
                    )
        if not metadata.get("reference_model"):
            raise ValueError("TailSFT reference metadata must identify the initial model")
        self.losses = np.load(root / "initial_losses.npy", mmap_mode="r", allow_pickle=False)
        self.lengths = np.load(root / "target_counts.npy", mmap_mode="r", allow_pickle=False)
        if self.losses.ndim != 1 or self.lengths.shape != self.losses.shape:
            raise ValueError("TailSFT reference scores must contain aligned one-dimensional arrays")
        if len(self.losses) != metadata.get("rows"):
            raise ValueError("TailSFT reference row count does not match metadata")
        if not np.isfinite(self.losses).all() or (self.lengths <= 0).any():
            raise ValueError("TailSFT reference contains invalid losses or empty targets")


def sequence_losses(
    losses: torch.Tensor, mask: torch.Tensor, boundaries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum differentiable losses and count target tokens within each packed segment."""
    values = losses.reshape(-1).float()
    mask = mask.reshape(-1).float()
    boundaries = boundaries.to(device=values.device, dtype=torch.long)
    if boundaries.ndim != 1 or boundaries.numel() < 2:
        raise ValueError("TailSFT needs a one-dimensional sequence boundary vector")
    if boundaries[0] != 0 or boundaries[-1] != values.numel() or (boundaries.diff() <= 0).any():
        raise ValueError("TailSFT sequence boundaries do not cover the token loss tensor")
    # Independent segment reductions avoid cancellation from subtracting large
    # pack-wide prefix sums. Accumulate NLLs in float64, then round sequence
    # means to float32 like the immutable reference cache.
    lengths = boundaries.diff()
    return (
        torch.segment_reduce(values.double() * mask, "sum", lengths=lengths),
        torch.segment_reduce(mask, "sum", lengths=lengths),
    )


def gather_variable(
    values: torch.Tensor, group: dist.ProcessGroup | None
) -> tuple[torch.Tensor, int]:
    """Gather different numbers of conversations per DP rank in rank order."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return values, 0
    world_size = dist.get_world_size(group)
    size = torch.tensor([values.numel()], device=values.device, dtype=torch.long)
    sizes = [torch.empty_like(size) for _ in range(world_size)]
    dist.all_gather(sizes, size, group=group)
    lengths = [int(item.item()) for item in sizes]
    padded = values.new_zeros(max(lengths))
    padded[: values.numel()] = values
    gathered = [torch.empty_like(padded) for _ in lengths]
    dist.all_gather(gathered, padded, group=group)
    rank = dist.get_rank(group)
    return torch.cat([item[:length] for item, length in zip(gathered, lengths)]), sum(
        lengths[:rank]
    )


def filtered_loss(
    losses: torch.Tensor,
    mask: torch.Tensor,
    boundaries: torch.Tensor,
    initial_losses: torch.Tensor,
    fraction: float,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return the selection-batch objective and reporting totals.

    The scalar is multiplied by DP size. With --calculate-per-token-loss we
    return one normalization unit per microbatch/rank, so Megatron's existing
    gradient finalizer divides by DP size times the accumulation count. This
    yields HF's average of selection-batch token means, even when survivor
    counts differ between accumulation steps. No core gradient code is changed.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("TailSFT drop fraction must be in [0, 1]")
    sums, counts = sequence_losses(losses, mask, boundaries)
    initial_losses = initial_losses.to(device=sums.device, dtype=torch.float32).reshape(-1)
    if initial_losses.shape != sums.shape:
        raise ValueError("TailSFT needs one reference loss per segment, including padding")
    eligible = counts > 0
    margins = (sums.detach()[eligible] / counts[eligible]).float() - initial_losses[eligible]
    margins, offset = gather_variable(margins, group)
    if not margins.numel() or not torch.isfinite(margins).all():
        raise ValueError("TailSFT selection batch has no valid targets or non-finite margins")
    keep = torch.ones_like(margins, dtype=torch.bool)
    # Stable ties preserve rank/sequence order on every replica.
    dropped = min(round(margins.numel() * fraction), margins.numel() - 1)
    keep[torch.argsort(margins, stable=True)[:dropped]] = False
    local_keep = torch.zeros_like(eligible)
    local_keep[eligible] = keep[offset : offset + int(eligible.sum().item())]
    numerator = (sums * local_keep).sum()
    token_count = (counts * local_keep).sum().detach()
    global_count = token_count.clone()
    world_size = 1
    if dist.is_initialized():
        dist.all_reduce(global_count, group=group)
        world_size = dist.get_world_size(group)
    scalar = (numerator * world_size / global_count).float()
    # Reports use local totals so the regular logger can sum across DP.
    report = {
        "lm loss": torch.stack((numerator.detach(), token_count)),
        "tail sft dropped fraction": torch.stack(
            ((eligible & ~local_keep).sum(), eligible.sum())
        ).float(),
    }
    return scalar, torch.ones((), device=sums.device, dtype=torch.int), report
