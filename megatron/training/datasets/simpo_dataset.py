# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.low_level_dataset import LowLevelDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.training import get_args, get_tokenizer

IGNORE_INDEX = -100

class SimPODataset(MegatronDataset):
    """The dataset used during SimPO (Simple Preference Optimization)"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, config)
        self.tokenizer = get_tokenizer()

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
        if pack_factor is None:
            pack_factor = max(1, pack_length // 1024)

        base_sample_idx = idx * pack_factor if pack_samples else idx
        curr_idx_offset = 0

        if len(self.indices) == 0:
            raise ValueError(f"SimPODataset received empty indices for split {self.index_split}. "
                             f"Check your dataset size and --split argument.")

        while len(pack_tokens) < pack_length + 1:
            sample_idx = int(self.indices[(base_sample_idx + curr_idx_offset) % len(self.indices)])
            # Expecting a list of TWO conversations: [chosen, rejected]
            row = self.dataset[sample_idx]
            if "conversations" in row:
                conversations_pair = row["conversations"]
            elif "messages" in row:
                conversations_pair = row["messages"]
            else:
                # Unexpected format
                curr_idx_offset += 1
                continue

            if not isinstance(conversations_pair, list) or len(conversations_pair) != 2:
                # Skips if not a pair
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
                
                # Positions restart for each conversation to keep them independent within the packed buffer
                start_pos = 0
                
                temp_tokens.extend(tokens_list)
                temp_targets.extend(targets_list)
                temp_positions.extend(range(start_pos, start_pos + len(tokens_list)))
                temp_cu_seqlens.append(len(pack_tokens) + len(temp_tokens))

            # Check if this pair fits in the remaining space
            if len(pack_tokens) + len(temp_tokens) <= pack_length + 1:
                pack_tokens.extend(temp_tokens)
                pack_targets.extend(temp_targets)
                pack_positions.extend(temp_positions)
                if pack_samples:
                    cu_seqlens.extend(temp_cu_seqlens)
                curr_idx_offset += 1
            else:
                if len(pack_tokens) == 0:
                    # Even empty buffer can't fit this pair. Skip to avoid infinite loop.
                    from megatron.training import print_rank_0
                    print_rank_0(f"WARNING: SimPO pair at index {sample_idx} is too long ({len(temp_tokens)} tokens) "
                                 f"for pack_length {pack_length}. Skipping entire pair.")
                    curr_idx_offset += 1
                    continue
                else:
                    # Buffer has data, this pair belongs in the next one.
                    break

            if not pack_samples:
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
        
        # cu_seqlens and max_seqlen for THD
        if pack_samples:
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
