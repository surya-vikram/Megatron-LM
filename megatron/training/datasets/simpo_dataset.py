# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.training import get_args

IGNORE_INDEX = -100

class SimPOLowLevelDataset:
    """The low-level dataset loading jsonl data for SimPO
    
    Each line of the jsonl must have key "messages" (List[List[Dict]]),
    which contains exactly TWO conversations: [chosen_conversation, rejected_conversation].
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SimPODataset requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx]["messages"]

class SimPODataset(MegatronDataset):
    """The dataset used during SimPO (Simple Preference Optimization)"""

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

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SimPOLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length
        args = get_args()

        pack_samples = getattr(args, "pack_samples", False)
        # We strictly require pack_samples for SimPO efficiency.
        # Fallback to standard 1 if not provided, but train.sh should enforce it.
        pack_factor = getattr(args, "pack_factor", max(1, pack_length // 1024))

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1]+1, positions[-1]+1+pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        pad = tokenizer.pad

        base_sample_idx = idx * pack_factor if pack_samples else idx
        curr_idx_offset = 0

        while len(pack_tokens) < pack_length + 1:
            sample_idx = int(self.indices[(base_sample_idx + curr_idx_offset) % len(self.indices)])
            # Expecting a list of TWO conversations: [chosen, rejected]
            conversations_pair = self.dataset[sample_idx]
            
            if len(conversations_pair) != 2:
                raise ValueError(f"SimPO expects exactly two conversations per line (chosen, rejected), found {len(conversations_pair)}")

            # Process both chosen and rejected to see if they fit together
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
                
                temp_tokens.extend(tokens_list)
                temp_targets.extend(targets_list)
                
                # Positions restart for the next conversation to make them independent
                if len(temp_positions) == 0:
                    start_pos = 0 if len(pack_positions) == 0 else pack_positions[-1] + 1
                else:
                    start_pos = temp_positions[-1] + 1
                temp_positions.extend(range(start_pos, start_pos + len(tokens_list)))
                
                temp_cu_seqlens.append(len(pack_tokens) + len(temp_tokens))

            # Check if adding BOTH fits in the remaining space
            if len(pack_tokens) + len(temp_tokens) <= pack_length + 1:
                # It fits entirely
                pack_tokens.extend(temp_tokens)
                pack_targets.extend(temp_targets)
                pack_positions.extend(temp_positions)
                cu_seqlens.extend(temp_cu_seqlens)
            else:
                # It doesn't fit together. 
                # If we haven't added anything yet, it means the pair is larger than the entire pack_length.
                # In this case, we MUST truncate it, but we truncate the pair together symmetrically if possible,
                # or just truncate the end. To keep it simple, we truncate the rejected one if it exceeds.
                if len(pack_tokens) == 0:
                    max_body = pack_length
                    pack_tokens = temp_tokens[:max_body]
                    pack_targets = temp_targets[:max_body]
                    pack_tokens.append(pad)
                    pack_targets.append(pad)
                    pack_positions = temp_positions[:pack_length+1]
                    
                    # Fix cu_seqlens. temp_cu_seqlens has 2 elements: end of chosen, end of rejected.
                    if temp_cu_seqlens[0] > pack_length:
                        # Chosen itself is larger than pack_length! Truncate chosen, discard rejected.
                        # (Not ideal for SimPO, but handled robustly)
                        cu_seqlens.append(pack_length)
                    else:
                        cu_seqlens.append(temp_cu_seqlens[0])
                        cu_seqlens.append(pack_length)
                    break
                else:
                    # We already have some pairs in this bin. We should skip this pair and just pad the rest.
                    break

            if len(pack_tokens) >= pack_length + 1 or not pack_samples:
                break

            curr_idx_offset += 1
            if curr_idx_offset >= pack_factor:
                break

        # Handle remaining padding
        if len(pack_tokens) < pack_length + 1:
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            
            if pack_samples:
                cu_seqlens.append(len(pack_tokens) - 1)
            else:
                cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        shifted_targets = torch.tensor(pack_targets[1:], dtype=torch.int64)
        loss_mask[shifted_targets == pad] = 0.0
        loss_mask[shifted_targets == IGNORE_INDEX] = 0.0

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()

        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
