# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.training import get_args
from megatron.training.datasets.chat_packing import (
    IndexedJsonlDataset,
    build_pack_index,
    fingerprint_tokenizer_path,
    load_pack_lengths,
    validate_metadata_source,
)

IGNORE_INDEX = -100

# Only emit each data warning once per process lifetime.
_warned_oversized = set()
_warned_malformed = set()


def validate_chat_messages(messages):
    """Return an error string for malformed chat messages, otherwise None."""
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list"

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            return f"message {message_index} must be an object"
        if message.get("role") not in {"system", "user", "assistant"}:
            return f"message {message_index} has an unsupported role"
        if not isinstance(message.get("content"), str):
            return f"message {message_index} content must be a string"

    return None


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


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx].get("messages")


def get_pack_metadata_path(args: Any, dataset_path: Optional[str]) -> Optional[str]:
    """Select training or validation packing metadata for a JSONL path."""
    valid_paths = [str(path) for path in (getattr(args, "valid_data_path", None) or [])]
    if dataset_path is not None and str(dataset_path) in valid_paths:
        return getattr(args, "valid_pack_metadata_path", None)
    return getattr(args, "pack_metadata_path", None)


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

        # ── Running stats (accumulated across __getitem__ calls on this worker) ──
        self._stats = {
            "steps":              0,
            "total_packed":       0,   # total conversations packed
            "total_active_tok":   0,   # tokens contributing to loss
            "total_pad_tok":      0,   # padding tokens
            "total_tok":          0,   # total tokens (= active + pad + prompt)
            "skipped_oversized":  0,   # conversations skipped because they exceed seq_len
            "skipped_malformed":  0,   # rows skipped because they have invalid messages
        }

        args = get_args()
        pack_samples = getattr(args, "pack_samples", False)
        metadata_path = get_pack_metadata_path(args, dataset_path)
        prompt_format = getattr(args, "sft_tokenizer_prompt_format", "default")
        if pack_samples and prompt_format == "chimera" and metadata_path is None:
            raise ValueError(
                "packed Chimera SFT requires --pack-metadata-path produced by "
                "examples/chimera/prepare_chat_data.py"
            )
        indexed_packing = pack_samples and metadata_path is not None
        self._pack_samples = indexed_packing
        self._pack_index = None
        if indexed_packing:
            context_parallel_size = getattr(config, "context_parallel_size", 1)
            if context_parallel_size > 1:
                raise NotImplementedError(
                    "packed Chimera SFT supports --context-parallel-size 1 only"
                )
            lengths, metadata = load_pack_lengths(
                metadata_path, expected_mode="sft", expected_rows=len(self.dataset)
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
            raise ValueError(f"SFT dataset {dataset_path!r} contains no rows")
        if self.indices is None or len(self.indices) == 0:
            raise ValueError(
                f"SFT dataset split {index_split.name!r} for {dataset_path!r} contains no rows"
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
                f"[SFTDataset] split={index_split.name} path={dataset_path} "
                f"physical_rows={len(self.dataset)} split_rows={len(self.indices)} "
                f"requested_samples={requested} pack_samples={pack_samples}{pack_summary}",
                flush=True,
            )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        args = get_args()
        metadata_path = get_pack_metadata_path(args, dataset_path)
        if getattr(args, "pack_samples", False) and metadata_path:
            return IndexedJsonlDataset(dataset_path, metadata_path, field="messages")
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self.num_samples
        if self._pack_index is not None:
            return len(self._pack_index)
        return len(self.indices)

    def _get_packed_item(self, idx: int) -> Dict[str, Any]:
        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length
        pad = tokenizer.pad
        epoch_pack_index = idx % len(self._pack_index)
        row_indices = self._pack_index.rows_for_pack(epoch_pack_index)

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]

        for row_index_value in row_indices:
            row_index = int(row_index_value)
            conversation = self.dataset[row_index]
            validation_error = validate_chat_messages(conversation)
            if validation_error is not None:
                raise RuntimeError(
                    f"SFT row {row_index} became invalid after packing metadata was built: "
                    f"{validation_error}"
                )
            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )
            sequence_tokens = tokens[:-1].tolist()
            sequence_targets = targets[1:].tolist()
            expected_length = int(self._pack_lengths[row_index])
            if len(sequence_tokens) != expected_length:
                raise RuntimeError(
                    f"SFT row {row_index} tokenized to {len(sequence_tokens)} positions, "
                    f"but packing metadata records {expected_length}; rebuild the metadata"
                )

            pack_tokens.extend(sequence_tokens)
            pack_targets.extend(sequence_targets)
            pack_positions.extend(range(len(sequence_tokens)))
            cu_seqlens.append(len(pack_tokens))

        real_length = len(pack_tokens)
        padding_length = pack_length - real_length
        if padding_length < 0:
            raise RuntimeError(
                f"SFT pack {epoch_pack_index} exceeds capacity: {real_length} > {pack_length}"
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
        }

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            # Whenever we see a new system message, start a new conversation
            if msg["role"] == "system":
                if current:  # If previously accumulating a conversation, then store it
                    split_conversations.append(current)
                current = [msg]  # Then start the new conversation
            else:
                current.append(msg) # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        if getattr(self, "_pack_samples", False):
            return self._get_packed_item(idx)

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length
        args = get_args()

        # Feature flags
        pack_samples    = getattr(args, "pack_samples", False)
        pack_factor     = getattr(args, "pack_factor", None)
        debug_dataset   = getattr(args, "debug_dataset", False)
        log_stats       = getattr(args, "log_dataset_stats", False)
        is_rank_0       = (os.environ.get("RANK", "0") == "0")

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1]+1, positions[-1]+1+pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        eod = tokenizer.eod
        pad = tokenizer.pad

        # ── Per-step debug counters ──
        step_packed      = 0
        step_skipped     = 0
        step_malformed   = 0

        # Deterministic, non-overlapping starting sample mapping
        if pack_samples:
            stride = pack_factor if pack_factor is not None else 1
            base_sample_idx = idx * stride
        else:
            base_sample_idx = idx

        curr_idx_offset = 0
        max_rows_to_scan = len(self.indices)
        if pack_samples and pack_factor is not None:
            max_rows_to_scan = min(max_rows_to_scan, pack_factor)
        while curr_idx_offset < max_rows_to_scan and len(pack_tokens) < pack_length + 1:
            sample_idx = int(self.indices[(base_sample_idx + curr_idx_offset) % len(self.indices)])
            merged_conversations = self.dataset[sample_idx]
            validation_error = validate_chat_messages(merged_conversations)
            if validation_error is not None:
                self._stats["skipped_malformed"] += 1
                step_malformed += 1
                if is_rank_0 and sample_idx not in _warned_malformed:
                    _warned_malformed.add(sample_idx)
                    print(
                        f"[SFTDataset][WARN] Permanently skipping sample idx={sample_idx}: "
                        f"{validation_error}.",
                        flush=True,
                    )
                curr_idx_offset += 1
                continue

            split_conversations = self._split_conversations(merged_conversations)
            should_break_outer = False

            for conversation in split_conversations:
                tokens, targets = tokenizer.tokenize_conversation(
                    conversation, return_target=True, add_generation_prompt=False
                )

                tokens_list = tokens.tolist()
                targets_list = targets.tolist()

                if not any(target != IGNORE_INDEX and target != pad for target in targets_list):
                    self._stats["skipped_malformed"] += 1
                    step_malformed += 1
                    if is_rank_0 and sample_idx not in _warned_malformed:
                        _warned_malformed.add(sample_idx)
                        print(
                            f"[SFTDataset][WARN] Permanently skipping sample idx={sample_idx}: "
                            "conversation contains no trainable assistant tokens.",
                            flush=True,
                        )
                    continue

                # Strictly pack ONLY if the entire conversation fits in the remaining space
                if len(pack_tokens) + len(tokens_list) <= pack_length + 1:
                    pack_tokens.extend(tokens_list)
                    pack_targets.extend(targets_list)
                    assert not self.config.reset_position_ids
                    pack_positions.extend(range(len(tokens_list)))
                    cu_seqlens.append(len(pack_tokens))
                    step_packed += 1
                else:
                    if len(pack_tokens) == 0:
                        # Oversized single conversation: skip it entirely (same as SimPO).
                        # Never truncate mid-turn — move to the next sample.
                        step_skipped += 1
                        self._stats["skipped_oversized"] += 1
                        if is_rank_0 and sample_idx not in _warned_oversized:
                            _warned_oversized.add(sample_idx)
                            print(
                                f"[SFTDataset][WARN] Sample idx={sample_idx} "
                                f"contains a conversation with {len(tokens_list)} tokens, "
                                f"exceeding the maximum capacity of {pack_length + 1}. "
                                "Permanently skipping it.",
                                flush=True,
                            )
                    else:
                        # Leave this conversation for the next packed step
                        should_break_outer = True
                    break

            curr_idx_offset += 1
            if should_break_outer or len(pack_tokens) >= pack_length + 1:
                break
            if not pack_samples and pack_tokens:
                break

        if not pack_tokens:
            raise RuntimeError(
                f"[SFTDataset] Could not construct sample idx={idx} after scanning "
                f"{curr_idx_offset} row(s) from {self.dataset_path!r}: "
                f"malformed={step_malformed}, oversized={step_skipped}. "
                f"No conversation fits sequence length {pack_length}."
            )

        # Handle remaining padding if under-filled
        if len(pack_tokens) < pack_length + 1:
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            if pack_samples:
                # CRITICAL: Append padding as an isolated dummy sequence boundary.
                # The real samples do not attend to it, saving quadratic compute!
                cu_seqlens.append(len(pack_tokens) - 1)
            else:
                # Merge padding into last sequence (default behavior)
                cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Align and convert to tensors
        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask: zero out pad and prompt (IGNORE_INDEX) positions
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        shifted_targets = torch.tensor(pack_targets[1:], dtype=torch.int64)
        loss_mask[shifted_targets == pad] = 0.0
        loss_mask[shifted_targets == IGNORE_INDEX] = 0.0

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask

        assert len(cu_seqlens) >= 2
        if pack_samples and cu_seqlens:
            # Exact-fit packed conversations can leave the terminal boundary at
            # pack_length + 1 even though the model consumes pack_tokens[:-1].
            # Clamp the tail so cu_seqlens always sums to the actual input len.
            cu_seqlens[-1] = min(cu_seqlens[-1], pack_length)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()

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
            seqs_in_pack = len(cu_seqlens) - 1
            print(
                f"[SFTDataset][DEBUG] step={self._stats['steps']:>6d} | "
                f"idx={idx:>6d} | packed={step_packed} conv(s) | "
                f"skipped_oversized={step_skipped} | skipped_malformed={step_malformed} | "
                f"seqs_in_pack={seqs_in_pack} | "
                f"active_tok={active_tok} | pad_tok={pad_tok} | "
                f"utilization={utilization:.1f}% | "
                f"cu_seqlens={cu_seqlens.tolist()}"
            )

        # ── Aggregate stats log (every 100 steps, rank 0 only) ──
        if log_stats and is_rank_0 and self._stats["steps"] % 100 == 0:
            s = self._stats
            avg_packed    = s["total_packed"]     / s["steps"]
            avg_active    = s["total_active_tok"] / s["steps"]
            avg_pad       = s["total_pad_tok"]    / s["steps"]
            avg_util      = 100.0 * s["total_active_tok"] / s["total_tok"] if s["total_tok"] > 0 else 0.0
            print(
                f"[SFTDataset][STATS] steps={s['steps']} | "
                f"avg_packed={avg_packed:.2f} conv/step | "
                f"avg_active_tok={avg_active:.1f} | "
                f"avg_pad_tok={avg_pad:.1f} | "
                f"utilization={avg_util:.1f}% | "
                f"total_skipped_oversized={s['skipped_oversized']} | "
                f"total_skipped_malformed={s['skipped_malformed']}"
            )

        ret = {
            'tokens':      input_ids,
            'labels':      labels,
            'loss_mask':   loss_mask,
            'position_ids': position_ids,
        }
        if pack_samples:
            ret['cu_seqlens'] = cu_seqlens
            ret['max_seqlen'] = max_seqlen
        return ret
