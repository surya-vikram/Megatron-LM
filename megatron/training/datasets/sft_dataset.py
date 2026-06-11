# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os
import atexit, json
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.training import get_args

IGNORE_INDEX = -100

# Only emit each oversized-sample warning once per process lifetime
_warned_oversized = set()


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
        return self.dataset[idx]["messages"]


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
        }

        args = get_args()
        if getattr(args, "pack_samples", False):
            self.collate_fn = PackSamplesCollator()

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

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

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length
        args = get_args()

        # Feature flags
        pack_samples    = getattr(args, "pack_samples", False)
        pack_factor     = getattr(args, "pack_factor", None)
        debug_dataset   = getattr(args, "debug_dataset", False)
        log_stats       = getattr(args, "log_dataset_stats", False)
        warn_oversized  = getattr(args, "warn_oversized_samples", False)
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

        # Deterministic, non-overlapping starting sample mapping
        if pack_samples:
            stride = pack_factor if pack_factor is not None else 1
            base_sample_idx = idx * stride
        else:
            base_sample_idx = idx

        should_break_outer = False
        curr_idx_offset = 0
        while len(pack_tokens) < pack_length + 1:
            sample_idx = int(self.indices[(base_sample_idx + curr_idx_offset) % len(self.indices)])
            merged_conversations = self.dataset[sample_idx]
            split_conversations = self._split_conversations(merged_conversations)

            for conversation in split_conversations:
                tokens, targets = tokenizer.tokenize_conversation(
                    conversation, return_target=True, add_generation_prompt=False
                )

                tokens_list = tokens.tolist()
                targets_list = targets.tolist()

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
                        if warn_oversized and sample_idx not in _warned_oversized:
                            _warned_oversized.add(sample_idx)
                            print(
                                f"[SFTDataset][WARN] Sample idx={sample_idx} "
                                f"({len(tokens_list)} tokens) exceeds seq_len={pack_length}. "
                                f"Skipping (will appear in the next available step)."
                            )
                    else:
                        # Leave this conversation for the next packed step
                        should_break_outer = True
                    break

            if should_break_outer or len(pack_tokens) >= pack_length + 1 or not pack_samples:
                break

            curr_idx_offset += 1
            # Bound packing range ONLY if a fixed pack_factor is explicitly supplied by the user
            if pack_factor is not None and curr_idx_offset >= pack_factor:
                break

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
                f"skipped={step_skipped} | seqs_in_pack={seqs_in_pack} | "
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
                f"total_skipped_oversized={s['skipped_oversized']}"
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
