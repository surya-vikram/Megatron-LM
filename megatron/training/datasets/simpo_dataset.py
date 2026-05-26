# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.training import get_args, get_tokenizer

IGNORE_INDEX = -100

class JsonlLowLevelDataset:
    """A simple low-level dataset for JSONL files."""
    def __init__(self, dataset_path: str):
        self.samples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
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

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: Any) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> Any:
        return JsonlLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        args = get_args()
        tokenizer = self.tokenizer
        pad = tokenizer.pad
        pack_length = self.config.sequence_length

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        max_seqlen = 0

        pack_samples = getattr(args, "pack_samples", False)
        pack_factor = getattr(args, "pack_factor", None)

        stride = pack_factor if pack_factor is not None else 1
        base_sample_idx = idx * stride if pack_samples else idx
        curr_idx_offset = 0

        if len(self.indices) == 0:
            return {} # Should not happen with correct split

        while len(pack_tokens) < pack_length + 1:
            sample_idx = int(self.indices[(base_sample_idx + curr_idx_offset) % len(self.indices)])
            row = self.dataset[sample_idx]
            
            if "conversations" in row:
                conversations_pair = row["conversations"]
            elif "messages" in row:
                conversations_pair = row["messages"]
            else:
                curr_idx_offset += 1
                continue

            if not isinstance(conversations_pair, list) or len(conversations_pair) != 2:
                curr_idx_offset += 1
                continue

            temp_tokens = []
            temp_targets = []
            temp_positions = []
            temp_cu_seqlens = []
            
            for conv in conversations_pair:
                tokens, targets = tokenizer.tokenize_conversation(
                    conv, return_target=True, add_generation_prompt=False
                )
                tokens_list = tokens.tolist()
                targets_list = targets.tolist()
                
                start_pos = 0
                temp_tokens.extend(tokens_list)
                temp_targets.extend(targets_list)
                temp_positions.extend(range(start_pos, start_pos + len(tokens_list)))
                temp_cu_seqlens.append(len(pack_tokens) + len(temp_tokens))

            # Strictly skip if it doesn't fit
            if len(pack_tokens) + len(temp_tokens) <= pack_length + 1:
                pack_tokens.extend(temp_tokens)
                pack_targets.extend(temp_targets)
                pack_positions.extend(temp_positions)
                if pack_samples:
                    cu_seqlens.extend(temp_cu_seqlens)
                curr_idx_offset += 1
            else:
                if len(pack_tokens) == 0:
                    # Skip oversized pair
                    curr_idx_offset += 1
                    continue
                else:
                    break

            if not pack_samples:
                break

            # Bound packing range ONLY if a fixed pack_factor is explicitly supplied by the user
            if pack_factor is not None and curr_idx_offset >= pack_factor:
                break

        # Terminal Padding
        if len(pack_tokens) < pack_length + 1:
            pad_len = pack_length + 1 - len(pack_tokens)
            last_pos = pack_positions[-1] if pack_positions else -1
            pack_tokens.extend([pad] * pad_len)
            pack_targets.extend([IGNORE_INDEX] * pad_len)
            pack_positions.extend(range(last_pos + 1, last_pos + 1 + pad_len))
            if pack_samples:
                cu_seqlens.append(len(pack_tokens) - 1)

        input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)
        
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        shifted_targets = torch.tensor(pack_targets[1:], dtype=torch.int64)
        loss_mask[shifted_targets == pad] = 0.0
        loss_mask[shifted_targets == IGNORE_INDEX] = 0.0
        
        if pack_samples:
            if cu_seqlens:
                cu_seqlens[-1] = min(cu_seqlens[-1], pack_length)
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        else:
            cu_seqlens = None
            max_seqlen = pack_length

        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
