"""Validation-only Chimera entrypoint that audits actual batches and TailSFT losses.

Set TAIL_SFT_AUDIT_DIR and launch instead of pretrain_chimera.py. This adds
expensive independent checks and tensor captures; it is not a training feature.
"""

import importlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/chimera"))

import pretrain_gpt
from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args

DIRECTORY = Path(os.environ["TAIL_SFT_AUDIT_DIR"])
DIRECTORY.mkdir(parents=True, exist_ok=True)
STATE = {"batch": None, "logits": None, "calls": 0}
GET_BATCH = pretrain_gpt.get_batch
FORWARD = pretrain_gpt.forward_step
FILTER = pretrain_gpt.filtered_loss


def get_batch(*args, **kwargs):
    batch = GET_BATCH(*args, **kwargs)
    STATE["batch"] = batch
    return batch


def capture_logits(module, inputs, outputs):
    STATE["logits"] = outputs[0].detach().transpose(0, 1).contiguous()


def forward(data_iterator, model, *args, **kwargs):
    output_layer = get_attr_wrapped_model(model, "output_layer", allow_none=True)
    if output_layer is not None and not getattr(output_layer, "_tail_audit_hook", False):
        output_layer.register_forward_hook(capture_logits)
        output_layer._tail_audit_hook = True
    STATE["logits"] = None
    return FORWARD(data_iterator, model, *args, **kwargs)


def audit_loss(losses, mask, boundaries, initial_losses, fraction, group=None):
    scalar, units, report = FILTER(losses, mask, boundaries, initial_losses, fraction, group)
    args = get_args()
    rank = dist.get_rank()
    dp = dist.get_world_size(group)
    batch = STATE["batch"]
    tokens, labels, batch_mask, _, positions, packed = batch[:6]
    torch.testing.assert_close(mask, batch_mask)
    assert torch.all((mask == 0) | (mask == 1))
    assert torch.equal(mask.bool(), (labels != -100) & (labels != get_args()._audit_pad))
    boundary_values = boundaries.tolist()
    flat, weights = losses.flatten(), mask.flatten()
    sums, counts = [], []
    # Independent slice reductions, rather than the production prefix sums.
    for start, end in zip(boundary_values, boundary_values[1:]):
        sums.append((flat[start:end].double() * weights[start:end]).sum().item())
        counts.append(int(weights[start:end].sum().item()))
    local = [
        (
            index,
            float(np.float32(sums[index] / count) - np.float32(initial_losses[index].item())),
            count,
        )
        for index, count in enumerate(counts)
        if count
    ]
    gathered = [None] * dp
    dist.all_gather_object(gathered, local, group=group)
    candidates = [
        (r, index, margin, count)
        for r, items in enumerate(gathered)
        for index, margin, count in items
    ]
    rejected_count = min(round(len(candidates) * fraction), len(candidates) - 1)
    rejected = {
        (r, index)
        for r, index, _, _ in sorted(candidates, key=lambda item: item[2])[:rejected_count]
    }
    dp_rank = dist.get_rank(group)
    keep = [count > 0 and (dp_rank, index) not in rejected for index, count in enumerate(counts)]
    expected_count = sum(count for r, index, _, count in candidates if (r, index) not in rejected)
    expected_loss = (
        sum(value for value, retained in zip(sums, keep) if retained) * dp / expected_count
    )
    assert scalar.item() == pytest_approx(expected_loss), (scalar.item(), expected_loss)
    assert units.item() == 1
    expected_grad = torch.zeros_like(weights)
    for index, (start, end) in enumerate(zip(boundary_values, boundary_values[1:])):
        if keep[index]:
            expected_grad[start:end] = weights[start:end] * dp / expected_count
    grad = torch.autograd.grad(scalar, losses, retain_graph=True)[0]
    torch.testing.assert_close(grad.flatten(), expected_grad, atol=1e-6, rtol=1e-5)
    # Observe the backward executed by the actual Megatron pipeline too.
    losses.register_hook(lambda actual: check_backward(actual, expected_grad))
    expected_fraction = args.tail_sft_filter_fraction
    iteration = args.curr_iteration
    if args.tail_sft_filter_schedule == "ramp":
        expected_fraction *= min(max(iteration, 0) / max(args.train_iters - 1, 1), 1.0)
    assert abs(fraction - expected_fraction) < 1e-12
    logits = STATE["logits"]
    if args.use_linear_cross_entropy:
        logits = None  # The fused path does not materialize the actual logits.
    logit_error = None
    if logits is not None and args.tensor_model_parallel_size == 1:
        independent = F.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100, reduction="none"
        ).reshape_as(losses)
        error = (independent - losses).abs()[mask.bool()]
        logit_error = error.max().item()
        torch.testing.assert_close(
            independent[mask.bool()], losses[mask.bool()], atol=2e-4, rtol=2e-4
        )
    step = STATE["calls"]
    STATE["calls"] += 1
    record = dict(
        rank=rank,
        dp_rank=dp_rank,
        iteration=iteration,
        microbatch_call=step,
        schedule=args.tail_sft_filter_schedule,
        fraction=fraction,
        actual_rejected_fraction=rejected_count / len(candidates),
        boundaries=boundary_values,
        target_counts=counts,
        current_sums=sums,
        current_means=[value / count if count else None for value, count in zip(sums, counts)],
        initial_means=initial_losses.tolist(),
        gathered_candidates=candidates,
        retained=keep,
        rejected=sorted(rejected),
        global_survivor_tokens=expected_count,
        scalar=scalar.item(),
        expected_scalar=expected_loss,
        ce_max_error=logit_error,
        gradients_verified=True,
    )
    with (DIRECTORY / f"rank{rank}.jsonl").open("a") as writer:
        writer.write(json.dumps(record) + "\n")
    torch.save(
        dict(
            tokens=tokens,
            labels=labels,
            loss_mask=mask,
            positions=positions,
            boundaries=boundaries,
            initial_losses=initial_losses,
            current_losses=losses.detach(),
            loss_grad=grad,
            logits=logits if iteration in (0, 2) else None,
        ),
        DIRECTORY / f"rank{rank}_batch{step}.pt",
    )
    return scalar, units, report


def check_backward(actual, expected):
    torch.testing.assert_close(actual.flatten(), expected, atol=1e-6, rtol=1e-5)
    rank = dist.get_rank()
    with (DIRECTORY / f"rank{rank}_backward.jsonl").open("a") as writer:
        writer.write(
            json.dumps(
                dict(
                    call=STATE["calls"] - 1,
                    verified=True,
                    nonzero_tokens=int(actual.count_nonzero().item()),
                )
            )
            + "\n"
        )
    return actual


def pytest_approx(value):
    # pytest is optional in runtime environments.
    class Approx:
        def __eq__(self, other):
            return abs(other - value) <= 1e-4 + 1e-5 * abs(value)

    return Approx()


pretrain_gpt.get_batch = get_batch
pretrain_gpt.forward_step = forward
pretrain_gpt.filtered_loss = audit_loss

# Pad is fetched after global arguments/tokenizer initialization, inside the
# first forward; keep the actual tokenizer as the authority for target masks.
ORIGINAL_FORWARD = pretrain_gpt.forward_step


def initialize_audit(*args, **kwargs):
    from megatron.training import get_tokenizer

    get_args()._audit_pad = get_tokenizer().pad
    return ORIGINAL_FORWARD(*args, **kwargs)


pretrain_gpt.forward_step = initialize_audit

training = importlib.import_module("megatron.training.training")
FINALIZE = training.finalize_model_grads


def finalize(model, num_tokens=None, *args, **kwargs):
    before = int(num_tokens.item()) if num_tokens is not None else None
    result = FINALIZE(model, num_tokens, *args, **kwargs)
    from megatron.core.num_microbatches_calculator import get_num_microbatches

    active = pretrain_gpt.tail_sft_enabled(get_args())
    expected = get_num_microbatches() * parallel_state.get_data_parallel_world_size()
    after = int(num_tokens.item()) if num_tokens is not None else None
    if active:
        assert after == expected, (before, after, expected)
    probes = []
    for chunk in model:
        for name, parameter in chunk.named_parameters():
            gradient = getattr(parameter, "main_grad", None)
            if gradient is not None and parameter.numel() <= 4096:
                assert torch.isfinite(gradient).all(), name
                probes.append(dict(parameter=name, norm=gradient.float().norm().item()))
                if len(probes) == 3:
                    break
        if len(probes) == 3:
            break
    with (DIRECTORY / f"rank{dist.get_rank()}_finalizer.jsonl").open("a") as writer:
        writer.write(
            json.dumps(
                dict(
                    iteration=get_args().curr_iteration,
                    active=active,
                    local_units=before,
                    global_units=after,
                    expected_units=expected,
                    gradient_probes=probes,
                )
            )
            + "\n"
        )
    return result


training.finalize_model_grads = finalize
runpy.run_path(str(ROOT / "examples/chimera/pretrain_chimera.py"), run_name="__main__")
