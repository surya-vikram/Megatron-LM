# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.training import get_args, get_tokenizer
from megatron.training.datasets.chat_packing import (
    IndexedJsonlDataset,
    build_pack_index,
    fingerprint_tokenizer_path,
    load_pack_lengths,
    validate_metadata_source,
)
from megatron.training.datasets.sft_dataset import get_pack_metadata_path, validate_chat_messages

IGNORE_INDEX = -100

# Only emit each warning once per process lifetime
_warned_oversized  = set()
_warned_malformed  = set()


class PackSamplesCollator:
    def __call__(self, batch):
        if "real_length" not in batch[0]:
            tokens = torch.stack([item['tokens'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            loss_mask = torch.stack([item['loss_mask'] for item in batch])
            position_ids = torch.stack([item['position_ids'] for item in batch])
            seq_len = tokens.shape[1]
            boundaries = []
            for item_index, item in enumerate(batch):
                cu_seqlens = item['cu_seqlens']
                boundaries.append(
                    cu_seqlens if item_index == 0 else cu_seqlens[1:] + item_index * seq_len
                )
            batched_cu_seqlens = torch.cat(boundaries).unsqueeze(0)
            max_seqlen = torch.stack(
                [torch.as_tensor(item['max_seqlen']) for item in batch]
            ).max().view(1)
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "cu_seqlens": batched_cu_seqlens,
                "max_seqlen": max_seqlen,
            }

        seq_len = batch[0]['tokens'].numel()
        real_slices = {key: [] for key in ('tokens', 'labels', 'loss_mask', 'position_ids')}
        padding_slices = {key: [] for key in real_slices}
        boundaries = [0]
        cursor = 0

        # Keep every chosen/rejected segment first and move per-pack padding to the end.
        # This preserves even/odd SimPO pairing for micro-batch sizes greater than one.
        for item in batch:
            real_length = int(item['real_length'])
            for key in real_slices:
                real_slices[key].append(item[key][:real_length])
                if real_length < seq_len:
                    padding_slices[key].append(item[key][real_length:])
            real_boundaries = item['cu_seqlens'][: 2 * int(item['num_pairs']) + 1]
            for boundary in real_boundaries[1:]:
                boundaries.append(cursor + int(boundary))
            cursor += real_length

        for item in batch:
            padding_length = seq_len - int(item['real_length'])
            if padding_length:
                cursor += padding_length
                boundaries.append(cursor)

        tensors = {}
        for key in real_slices:
            pieces = real_slices[key] + padding_slices[key]
            tensors[key] = torch.cat(pieces).view(len(batch), seq_len)

        batched_cu_seqlens = torch.tensor(boundaries, dtype=torch.int32).unsqueeze(0)
        max_seqlen = (batched_cu_seqlens[0, 1:] - batched_cu_seqlens[0, :-1]).max().view(1)
        return {
            **tensors,
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
        metadata_path = get_pack_metadata_path(args, dataset_path)
        prompt_format = getattr(args, "sft_tokenizer_prompt_format", "default")
        if pack_samples and prompt_format == "chimera" and metadata_path is None:
            raise ValueError(
                "packed Chimera SimPO requires --pack-metadata-path produced by "
                "examples/chimera/prepare_chat_data.py"
            )
        indexed_packing = pack_samples and metadata_path is not None
        self._pack_samples = indexed_packing
        self._pack_index = None
        if (
            getattr(args, "simpo", False)
            and not pack_samples
            and getattr(args, "micro_batch_size", 1) != 1
        ):
            raise ValueError("Unpacked SimPO currently requires --micro-batch-size 1")
        if indexed_packing:
            context_parallel_size = getattr(config, "context_parallel_size", 1)
            if context_parallel_size > 1:
                raise NotImplementedError(
                    "packed Chimera SimPO supports --context-parallel-size 1 only"
                )
            lengths, metadata = load_pack_lengths(
                metadata_path, expected_mode="simpo", expected_rows=len(self.dataset)
            )
            validate_metadata_source(metadata, dataset_path)
            if metadata.get("prompt_format") != prompt_format:
                raise ValueError(
                    f"chat packing metadata prompt format is {metadata.get('prompt_format')!r}, "
                    f"expected {prompt_format!r}"
                )
            tokenizer_path = getattr(args, "tokenizer_model", None)
            if tokenizer_path and metadata.get("tokenizer_fingerprint") != fingerprint_tokenizer_path(
                tokenizer_path
            ):
                raise ValueError(
                    "chat packing metadata tokenizer does not match --tokenizer-model; "
                    "rerun examples/chimera/prepare_chat_data.py"
                )
            self._pack_lengths = lengths
            self._pack_index = build_pack_index(
                self.indices, self._pack_lengths, self.config.sequence_length
            )
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
            pack_summary = ""
            if self._pack_index is not None:
                utilization = (
                    100.0
                    * self._pack_index.packed_token_count
                    / (len(self._pack_index) * self.config.sequence_length)
                )
                pack_summary = (
                    f" epoch_packs={len(self._pack_index)} "
                    f"packed_rows={len(self._pack_index.row_indices)} "
                    f"invalid_rows={self._pack_index.invalid_row_count} "
                    f"oversized_rows={len(self._pack_index.oversized_row_indices)}"
                    f" token_utilization={utilization:.2f}%"
                )
            print(
                f"[SimPODataset] split={index_split.name} path={dataset_path} "
                f"physical_rows={len(self.dataset)} split_rows={len(self.indices)} "
                f"requested_samples={requested} pack_samples={pack_samples}{pack_summary}",
                flush=True,
            )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: Any) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> Any:
        args = get_args()
        metadata_path = get_pack_metadata_path(args, dataset_path)
        if getattr(args, "pack_samples", False) and metadata_path:
            return IndexedJsonlDataset(dataset_path, metadata_path)
        return JsonlLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self.num_samples
        if self._pack_index is not None:
            return len(self._pack_index)
        return len(self.indices)

    def _get_packed_item(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        pad = tokenizer.pad
        pack_length = self.config.sequence_length
        epoch_pack_index = idx % len(self._pack_index)
        row_indices = self._pack_index.rows_for_pack(epoch_pack_index)

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]

        for row_index_value in row_indices:
            row_index = int(row_index_value)
            row = self.dataset[row_index]
            if not isinstance(row, dict) or "chosen" not in row or "rejected" not in row:
                raise RuntimeError(
                    f"SimPO row {row_index} became invalid after packing metadata was built"
                )

            measured_pair_length = 0
            for name in ("chosen", "rejected"):
                conversation = row[name]
                validation_error = validate_chat_messages(conversation)
                if validation_error is not None:
                    raise RuntimeError(
                        f"SimPO row {row_index} {name} became invalid after packing metadata "
                        f"was built: {validation_error}"
                    )
                tokens, targets = tokenizer.tokenize_conversation(
                    conversation, return_target=True, add_generation_prompt=False
                )
                sequence_tokens = tokens[:-1].tolist()
                sequence_targets = targets[1:].tolist()
                measured_pair_length += len(sequence_tokens)
                pack_tokens.extend(sequence_tokens)
                pack_targets.extend(sequence_targets)
                pack_positions.extend(range(len(sequence_tokens)))
                cu_seqlens.append(len(pack_tokens))

            expected_pair_length = int(self._pack_lengths[row_index])
            if measured_pair_length != expected_pair_length:
                raise RuntimeError(
                    f"SimPO row {row_index} tokenized to {measured_pair_length} positions, "
                    f"but packing metadata records {expected_pair_length}; rebuild the metadata"
                )

        real_length = len(pack_tokens)
        padding_length = pack_length - real_length
        if padding_length < 0:
            raise RuntimeError(
                f"SimPO pack {epoch_pack_index} exceeds capacity: {real_length} > {pack_length}"
            )
        if padding_length:
            pack_tokens.extend([pad] * padding_length)
            pack_targets.extend([pad] * padding_length)
            pack_positions.extend(range(padding_length))
            cu_seqlens.append(pack_length)

        input_ids = torch.tensor(pack_tokens, dtype=torch.int64)
        labels = torch.tensor(pack_targets, dtype=torch.int64)
        position_ids = torch.tensor(pack_positions, dtype=torch.int64)
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0
        loss_mask[labels == IGNORE_INDEX] = 0.0
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        return {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "real_length": real_length,
            "num_pairs": len(row_indices),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if getattr(self, "_pack_samples", False):
            return self._get_packed_item(idx)

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
