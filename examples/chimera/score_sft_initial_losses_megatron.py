"""Record TailSFT reference NLLs using the actual Megatron checkpoint/packing path.

Invoke through sft.sh with TAIL_SFT_SCORE_OUTPUT set and TAIL_SFT=false.
Scoring supports DP with TP/PP/EP/CP=1 and creates no optimizer or training step.
"""

import json
import math
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from architecture_contract import validate_training_args
from pretrain_chimera import add_chimera_args, apply_chimera_yarn_args, chimera_builder

from megatron.core import parallel_state
from megatron.core.datasets.utils import Split
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank
from megatron.training.arguments import parse_and_validate_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.datasets.chat_packing import fingerprint_tokenizer_path
from megatron.training.datasets.sft_dataset import (
    PackSamplesCollator,
    SFTDataset,
    validate_chat_messages,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.tail_sft import file_digest
from megatron.training.training import get_model
from model_provider import model_provider
from pretrain_gpt import core_gpt_dataset_config_from_args


def add_args(parser):
    parser = add_chimera_args(parser)
    parser.add_argument("--tail-sft-score-output", required=True)
    parser.add_argument("--tail-sft-score-audit-dir")
    return parser


def main() -> None:
    args = apply_chimera_yarn_args(parse_and_validate_args(extra_args_provider=add_args))
    validate_training_args(args)
    if args.tail_sft or not args.sft or not args.load:
        raise ValueError("Reference scoring requires --sft, --load and disabled --tail-sft")
    if any(
        size != 1
        for size in (
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            args.expert_model_parallel_size,
            args.context_parallel_size,
        )
    ):
        raise ValueError("Native reference scoring supports DP with TP/PP/EP/CP=1")
    if len(args.train_data_path or []) != 1:
        raise ValueError("Reference scoring requires one training JSONL")
    output = Path(args.tail_sft_score_output)
    audit = Path(args.tail_sft_score_audit_dir) if args.tail_sft_score_audit_dir else None
    if output.exists() or (audit and audit.exists()):
        raise ValueError("Reference/audit output directories must be new")
    data_path = args.train_data_path[0]
    before = file_digest(data_path)
    # Materialize logits for optional audits; the CE objective is unchanged.
    args.use_linear_cross_entropy = False
    if args.pack_samples and args.position_embedding_type == "yarn":
        args.apply_rope_fusion = False
    initialize_megatron()
    models = get_model(
        partial(model_provider, chimera_builder), ModelType.encoder_or_decoder, wrap_with_ddp=False
    )
    load_checkpoint(models, None, None)
    model = models[0]
    model.train()  # Match the training forward kernels; gradients remain disabled.
    get_attr_wrapped_model(model, "config").moe_z_loss_coeff = None
    captured = {}
    if audit:
        audit.mkdir(parents=True, exist_ok=True)
        get_attr_wrapped_model(model, "output_layer").register_forward_hook(
            lambda module, inputs, outputs: captured.update(
                logits=outputs[0].detach().transpose(0, 1)
            )
        )
    config = core_gpt_dataset_config_from_args(args)
    low_level = SFTDataset.build_low_level_dataset(data_path, config)
    dataset = SFTDataset(low_level, data_path, np.arange(len(low_level)), None, Split.train, config)
    counts = []
    for row in range(len(low_level)):
        error = validate_chat_messages(low_level[row])
        if error:
            raise ValueError(f"Invalid row {row}: {error}")
        tokens, targets = config.tokenizer.tokenize_conversation(
            low_level[row], return_target=True, add_generation_prompt=False
        )
        count = int(np.count_nonzero((targets[1:] != -100) & (targets[1:] != config.tokenizer.pad)))
        if not count or len(tokens) - 1 > config.sequence_length:
            raise ValueError(f"Empty or oversized scoring row {row}")
        counts.append(count)
    if not args.pack_samples:
        # Use the strict TailSFT row path, avoiding ordinary SFT's row substitution.
        dataset._tail_reference = SimpleNamespace(
            losses=np.zeros(len(low_level)), lengths=np.array(counts)
        )
    number = len(dataset)
    rank = parallel_state.get_data_parallel_rank()
    world = parallel_state.get_data_parallel_world_size()
    local = []
    with torch.no_grad():
        for step in range(math.ceil(number / (world * args.micro_batch_size))):
            indices = [
                ((step * world + rank) * args.micro_batch_size + i) % number
                for i in range(args.micro_batch_size)
            ]
            items = [dataset[index] for index in indices]
            batch = PackSamplesCollator()(items) if args.pack_samples else default_collate(items)
            batch.pop("tail_sft_initial_losses", None)
            batch = {key: value.cuda() for key, value in batch.items()}
            if args.pack_samples:
                boundaries = batch.pop("cu_seqlens")[0]
                maximum = batch.pop("max_seqlen")
                for key in ("tokens", "labels", "loss_mask", "position_ids"):
                    batch[key] = batch[key].reshape(1, -1)
                batch, params = get_thd_batch_on_this_cp_rank(batch, boundaries, None, maximum)
                rows = []
                for index, item in zip(indices, items):
                    physical = list(dataset._pack_index.rows_for_pack(index))
                    rows.extend(physical)
                    if len(item["cu_seqlens"]) - 1 > len(physical):
                        rows.append(None)
            else:
                params = None
                boundaries = torch.arange(
                    0, batch["labels"].numel() + 1, args.seq_length, device="cuda"
                )
                rows = indices
            loss = (
                model(
                    batch["tokens"],
                    batch["position_ids"],
                    None,
                    labels=batch["labels"],
                    loss_mask=batch["loss_mask"],
                    packed_seq_params=params,
                )
                .flatten()
                .float()
            )
            mask = batch["loss_mask"].flatten().bool()
            for segment, (start, end) in enumerate(
                zip(boundaries[:-1].tolist(), boundaries[1:].tolist())
            ):
                row = rows[segment]
                if row is None:
                    continue
                active = mask[start:end]
                count = int(active.sum())
                assert count == counts[row]
                mean = loss[start:end][active].double().mean().item()
                local.append((int(row), mean, count))
                if audit:
                    tokens, targets = config.tokenizer.tokenize_conversation(
                        low_level[row], return_target=True, add_generation_prompt=False
                    )
                    length = len(tokens) - 1
                    torch.save(
                        dict(
                            inputs=torch.tensor(tokens[:-1]).unsqueeze(0),
                            labels=torch.tensor(targets[1:]),
                            mask=mask[start : start + length].cpu(),
                            logits=captured["logits"]
                            .reshape(-1, captured["logits"].shape[-1])[start : start + length]
                            .cpu(),
                            token_nll=(
                                loss[start : start + length] * mask[start : start + length]
                            ).cpu(),
                            mean_nll=mean,
                            target_count=count,
                        ),
                        audit / f"rank{rank}_row{int(row)}.pt",
                    )
    gathered = [None] * world
    dist.all_gather_object(gathered, local, group=parallel_state.get_data_parallel_group())
    if rank == 0:
        scores = np.full(len(low_level), np.nan, dtype=np.float32)
        for records in gathered:
            for row, score, count in records:
                if np.isfinite(scores[row]) and abs(float(scores[row]) - score) > 1e-4:
                    raise ValueError(f"Scoring batch changes NLL for repeated row {row}")
                scores[row] = score
        if not np.isfinite(scores).all() or before != file_digest(data_path):
            raise ValueError("Incomplete reference scores or changed dataset")
        output.mkdir(parents=True)
        np.save(output / "initial_losses.npy", scores)
        np.save(output / "target_counts.npy", np.array(counts, dtype=np.int64))
        metadata = dict(
            version=1,
            rows=len(scores),
            dataset_sha256=before,
            tokenizer_fingerprint=fingerprint_tokenizer_path(args.tokenizer_model),
            prompt_format=args.sft_tokenizer_prompt_format,
            target_mask="sft_tokenizer_all_assistant_turns",
            reference_model=str(Path(args.load).resolve()),
            backend="megatron",
            packing=args.pack_samples,
            sequence_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            apply_rope_fusion=args.apply_rope_fusion,
        )
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Recorded {len(scores)} native Megatron reference losses in {output}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
