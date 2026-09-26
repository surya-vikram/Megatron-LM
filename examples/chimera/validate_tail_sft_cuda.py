"""Validate TailSFT selection/gradients through NCCL before checkpoint smoke runs.

Run with: torchrun --standalone --nproc-per-node=2 examples/chimera/validate_tail_sft_cuda.py
This checks the collective/loss kernel path, not the complete model pipeline.
"""

import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        spec = importlib.util.spec_from_file_location(
            "tail_sft_math", Path(__file__).resolve().parents[2] / "megatron/training/tail_sft.py"
        )
        tail = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tail)
        # rank r has r+1 conversations, monotonically increasing global margins.
        offset = rank * (rank + 1) // 2
        factors = torch.arange(offset + 1, offset + rank + 2, device="cuda", dtype=torch.float32)
        total = world * (world + 1) // 2
        parameter = torch.tensor(2.0, device="cuda", requires_grad=True)
        for fraction in (0.0, 0.5, 1.0):
            parameter.grad = None
            objective, units, _ = tail.filtered_loss(
                parameter * factors,
                torch.ones_like(factors),
                torch.arange(len(factors) + 1, device="cuda"),
                torch.zeros_like(factors),
                fraction,
            )
            objective.backward()
            dist.all_reduce(parameter.grad)
            dropped = min(round(total * fraction), total - 1)
            expected = torch.arange(
                dropped + 1, total + 1, device="cuda", dtype=torch.float32
            ).mean()
            torch.testing.assert_close(parameter.grad / world, expected)
            assert units.item() == 1
        dist.barrier()
        if rank == 0:
            print(f"TailSFT CUDA/NCCL gradient checks passed on {world} ranks", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
