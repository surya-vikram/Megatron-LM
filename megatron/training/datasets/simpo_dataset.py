# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.training import get_args, get_tokenizer
from megatron.training.datasets.sft_dataset import validate_chat_messages

IGNORE_INDEX = -100

# Only emit each warning once per process lifetime
_warned_oversized  = set()
_warned_malformed  = set()


class PackSamplesCollator:
    def __call__(self, batch):
        tokens = torch.stack([item['tokens'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        loss_mask = torch.stack([item['loss_mask'] for item in batch])
        position_ids = torch.stack([item['position_ids'] for item in batch])
        
        seq_len = tokens.shape[1]
        cu_seqlens_list = []
        for i, item in enumerate(batch):
            cu_seqlens = item['cu_seqlens']
            if i > 0:
                shifted = cu_seqlens[1:] + i * seq_len
                cu_seqlens_list.append(shifted)
            else:
                cu_seqlens_list.append(cu_seqlens)
                
        batched_cu_seqlens = torch.cat(cu_seqlens_list).unsqueeze(0)
        max_seqlen = torch.stack([torch.as_tensor(item['max_seqlen']) for item in batch]).max().unsqueeze(0)
        
        return {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': batched_cu_seqlens,
            'max_seqlen': max_seqlen
        }


class JsonlLowLevelDataset:
    """A simple low-level dataset for JSONL files."""
    def __init__(self, dataset_path: str):
        self.samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in SimPO dataset {dataset_path!r} at line {line_number}"
                    ) from error
        if not self.samples:
            raise ValueError(f"SimPO dataset {dataset_path!r} contains no rows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SimPODataset(MegatronDataset):
    """The dataset used during SimPO (Simple Preference Optimization)"""

    def __init__(
        self,
        dataset: Any,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Any,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        self.tokenizer = get_tokenizer()

        # ── Running stats (accumulated across __getitem__ calls on this worker) ──
        self._stats = {
            "steps":              0,
            "total_packed":       0,   # total preference pairs packed
            "total_active_tok":   0,   # tokens contributing to loss
            "total_pad_tok":      0,   # padding tokens
            "total_tok":          0,   # total tokens per step
            "skipped_oversized":  0,   # pairs skipped for exceeding seq_len
            "skipped_malformed":  0,   # rows skipped for wrong format
        }

        args = get_args()
        pack_samples = getattr(args, "pack_samples", False)
        if (
            getattr(args, "simpo", False)
            and not pack_samples
            and getattr(args, "micro_batch_size", 1) != 1
        ):
            raise ValueError("Unpacked SimPO currently requires --micro-batch-size 1")
        if pack_samples:
            self.collate_fn = PackSamplesCollator()

        if len(self.dataset) == 0:
            raise ValueError(f"SimPO dataset {dataset_path!r} contains no rows")
        if self.indices is None or len(self.indices) == 0:
            raise ValueError(
                f"SimPO dataset split {index_split.name!r} for {dataset_path!r} contains no rows"
            )
        if os.environ.get("RANK", "0") == "0":
            requested = self.num_samples if self.num_samples is not None else len(self.indices)
            print(
                f"[SimPODataset] split={index_split.name} path={dataset_path} "
                f"physical_rows={len(self.dataset)} split_rows={len(self.indices)} "
                f"requested_samples={requested} pack_samples={pack_samples}",
                flush=True,
            )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: Any) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> Any:
        return JsonlLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples if self.num_samples is not None else len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        args = get_args()
        tokenizer = self.tokenizer
        pad = tokenizer.pad
        pack_length = self.config.sequence_length

        # Feature flags
        pack_samples    = getattr(args, "pack_samples", False)
        needs_cu_seqlens = pack_samples or getattr(args, "simpo", False)
        pack_factor     = getattr(args, "pack_factor", None)
        debug_dataset   = getattr(args, "debug_dataset", False)
        log_stats       = getattr(args, "log_dataset_stats", False)
        is_rank_0       = (os.environ.get("RANK", "0") == "0")

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        max_seqlen = 0

        # ── Per-step debug counters ──
        step_packed   = 0
        step_skipped  = 0
        step_malformed = 0

        stride = pack_factor if pack_factor is not None else 1
        base_sample_idx = idx * stride if pack_samples else idx
        curr_idx_offset = 0

        max_rows_to_scan = len(self.indices)
        if pack_samples and pack_factor is not None:
            max_rows_to_scan = min(max_rows_to_scan, pack_factor)

        while curr_idx_offset < max_rows_to_scan and len(pack_tokens) < pack_length + 1:
            sample_idx = int(self.indices[(base_sample_idx + curr_idx_offset) % len(self.indices)])
            row = self.dataset[sample_idx]

            if not isinstance(row, dict) or "chosen" not in row or "rejected" not in row:
                self._stats["skipped_malformed"] += 1
                step_malformed += 1
                if is_rank_0 and sample_idx not in _warned_malformed:
                    _warned_malformed.add(sample_idx)
                    print(
                        f"[SimPODataset][WARN] Permanently skipping sample idx={sample_idx}: "
                        "row must contain 'chosen' and 'rejected' message lists.",
                        flush=True,
                    )
                curr_idx_offset += 1
                continue

            conversations_pair = [row["chosen"], row["rejected"]]
            validation_errors = [
                validate_chat_messages(conversation) for conversation in conversations_pair
            ]
            if any(error is not None for error in validation_errors):
                self._stats["skipped_malformed"] += 1
                step_malformed += 1
                if is_rank_0 and sample_idx not in _warned_malformed:
                    _warned_malformed.add(sample_idx)
                    details = "; ".join(
                        f"{name}: {error}"
                        for name, error in zip(("chosen", "rejected"), validation_errors)
                        if error is not None
                    )
                    print(
                        f"[SimPODataset][WARN] Permanently skipping sample idx={sample_idx}: "
                        f"{details}.",
                        flush=True,
                    )
                curr_idx_offset += 1
                continue

            temp_tokens = []
            temp_targets = []
            temp_positions = []
            temp_cu_seqlens = []
            pair_has_trainable_targets = True

            for conv in conversations_pair:
                tokens, targets = tokenizer.tokenize_conversation(
                    conv, return_target=True, add_generation_prompt=False
                )
                tokens_list = tokens.tolist()
                targets_list = targets.tolist()
                pair_has_trainable_targets = pair_has_trainable_targets and any(
                    target != IGNORE_INDEX and target != pad for target in targets_list
                )

                start_pos = 0
                temp_tokens.extend(tokens_list)
                temp_targets.extend(targets_list)
                temp_positions.extend(range(start_pos, start_pos + len(tokens_list)))
                temp_cu_seqlens.append(len(pack_tokens) + len(temp_tokens))

            if not pair_has_trainable_targets:
                self._stats["skipped_malformed"] += 1
                step_malformed += 1
                if is_rank_0 and sample_idx not in _warned_malformed:
                    _warned_malformed.add(sample_idx)
                    print(
                        f"[SimPODataset][WARN] Permanently skipping sample idx={sample_idx}: "
                        "chosen and rejected must both contain trainable assistant tokens.",
                        flush=True,
                    )
                curr_idx_offset += 1
                continue

            # Strictly skip if the pair doesn't fit
            if len(pack_tokens) + len(temp_tokens) <= pack_length + 1:
                pack_tokens.extend(temp_tokens)
                pack_targets.extend(temp_targets)
                pack_positions.extend(temp_positions)
                if needs_cu_seqlens:
                    cu_seqlens.extend(temp_cu_seqlens)
                step_packed += 1
                curr_idx_offset += 1
            else:
                if len(pack_tokens) == 0:
                    # A pair that cannot fit in an empty sequence can never be trained.
                    self._stats["skipped_oversized"] += 1
                    step_skipped += 1
                    if is_rank_0 and sample_idx not in _warned_oversized:
                        _warned_oversized.add(sample_idx)
                        print(
                            f"[SimPODataset][WARN] Sample idx={sample_idx} "
                            f"contains a pair with {len(temp_tokens)} tokens, exceeding the "
                            f"maximum capacity of {pack_length + 1}. Permanently skipping it.",
                            flush=True,
                        )
                    curr_idx_offset += 1
                    continue
                else:
                    break

            if not pack_samples:
                break

        if not pack_tokens:
            raise RuntimeError(
                f"[SimPODataset] Could not construct sample idx={idx} after scanning "
                f"{curr_idx_offset} row(s) from {self.dataset_path!r}: "
                f"malformed={step_malformed}, oversized={step_skipped}. "
                f"No chosen/rejected pair fits sequence length {pack_length}."
            )

        # Terminal Padding
        if len(pack_tokens) < pack_length + 1:
            pad_len = pack_length + 1 - len(pack_tokens)
            last_pos = pack_positions[-1] if pack_positions else -1
            pack_tokens.extend([pad] * pad_len)
            pack_targets.extend([pad] * pad_len)
            pack_positions.extend(range(last_pos + 1, last_pos + 1 + pad_len))
            if needs_cu_seqlens:
                cu_seqlens.append(len(pack_tokens) - 1)

        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:],  dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        shifted_targets = torch.tensor(pack_targets[1:], dtype=torch.int64)
        loss_mask[shifted_targets == pad] = 0.0
        loss_mask[shifted_targets == IGNORE_INDEX] = 0.0

        if needs_cu_seqlens:
            if cu_seqlens:
                cu_seqlens[-1] = min(cu_seqlens[-1], pack_length)
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        else:
            cu_seqlens = None
            max_seqlen = pack_length

        # ── Stats accumulation ──
        active_tok = int(loss_mask.sum().item())
        pad_tok    = int((shifted_targets == pad).sum().item())
        total_tok  = pack_length

        self._stats["steps"]            += 1
        self._stats["total_packed"]     += step_packed
        self._stats["total_active_tok"] += active_tok
        self._stats["total_pad_tok"]    += pad_tok
        self._stats["total_tok"]        += total_tok

        # ── Per-step debug log (rank 0 only) ──
        if debug_dataset and is_rank_0:
            utilization = 100.0 * active_tok / total_tok if total_tok > 0 else 0.0
            seqs_in_pack = (len(cu_seqlens) - 1) if cu_seqlens is not None else 1
            print(
                f"[SimPODataset][DEBUG] step={self._stats['steps']:>6d} | "
                f"idx={idx:>6d} | packed={step_packed} pair(s) | "
                f"skipped_oversized={step_skipped} | skipped_malformed={step_malformed} | "
                f"seqs_in_pack={seqs_in_pack} | "
                f"active_tok={active_tok} | pad_tok={pad_tok} | "
                f"utilization={utilization:.1f}%"
            )

        # ── Aggregate stats log (every 100 steps, rank 0 only) ──
        if log_stats and is_rank_0 and self._stats["steps"] % 100 == 0:
            s = self._stats
            avg_packed = s["total_packed"]     / s["steps"]
            avg_active = s["total_active_tok"] / s["steps"]
            avg_pad    = s["total_pad_tok"]    / s["steps"]
            avg_util   = 100.0 * s["total_active_tok"] / s["total_tok"] if s["total_tok"] > 0 else 0.0
            print(
                f"[SimPODataset][STATS] steps={s['steps']} | "
                f"avg_packed={avg_packed:.2f} pair(s)/step | "
                f"avg_active_tok={avg_active:.1f} | "
                f"avg_pad_tok={avg_pad:.1f} | "
                f"utilization={avg_util:.1f}% | "
                f"total_skipped_oversized={s['skipped_oversized']} | "
                f"total_skipped_malformed={s['skipped_malformed']}"
            )

        return {
            'tokens':       input_ids,
            'labels':       labels,
            'loss_mask':    loss_mask,
            'position_ids': position_ids,
            'cu_seqlens':   cu_seqlens,
            'max_seqlen':   max_seqlen,
        }
