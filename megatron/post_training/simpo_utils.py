# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

def calculate_simpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    args
):
    """
    Computes the SimPO (Simple Preference Optimization) loss.
    
    Expects input tensors from a THD-packed format where sequences are independent,
    and chosen/rejected conversations appear as consecutive pairs.
    """
    # 1. Compute per-token log probabilities
    # We use vocab_parallel_cross_entropy to handle TP sharded logits correctly.
    
    # Flatten everything to 1D/2D base formats for THD consistency
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
    if labels.dim() == 2:
        labels = labels.view(-1)
    if loss_mask.dim() == 2:
        loss_mask = loss_mask.view(-1)
        
    # Handle ignore_index (-100) for vocab_parallel_cross_entropy
    labels_for_tp = labels.clone()
    labels_for_tp[labels == -100] = 0
    
    # Calculate negative log-likelihood (NLL)
    nll = vocab_parallel_cross_entropy(logits, labels_for_tp)
    
    # Strictly mask out NLL for padding/prompt tokens
    nll = nll * loss_mask
    per_token_logps = -nll
    
    # 2. Compute sequence-level average log probabilities (Length Normalization)
    seq_avg_logps = []
    pair_valid_masks = [] # Track which sequences are actually valid (mask sum > 0)
    chosen_sft_losses = []

    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i+1]

        seq_logps = per_token_logps[start_idx:end_idx]
        seq_mask = loss_mask[start_idx:end_idx]

        sum_mask = seq_mask.sum()
        if sum_mask > 0:
            avg_logp = seq_logps.sum() / sum_mask
            pair_valid_masks.append(True)
            
            # For Chosen sequences (even indices), we also collect the SFT loss component
            if i % 2 == 0:
                chosen_sft_losses.append(nll[start_idx:end_idx].sum() / sum_mask)
        else:
            # Dummy sequence or all-padding. Use -1e9 for metric but mark as invalid.
            avg_logp = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
            pair_valid_masks.append(False)

        seq_avg_logps.append(avg_logp)

    seq_avg_logps = torch.stack(seq_avg_logps)
    
    # 3. Group into Chosen and Rejected pairs
    # Since SimPODataset packs [chosen, rejected] pairs, chosen are even indices, rejected are odd indices.
    num_total_sequences = len(seq_avg_logps)
    num_possible_pairs = num_total_sequences // 2
    
    valid_chosen_logps = []
    valid_rejected_logps = []
    valid_chosen_sft_losses = []
    
    for p in range(num_possible_pairs):
        c_idx = 2 * p
        r_idx = 2 * p + 1
        if pair_valid_masks[c_idx] and pair_valid_masks[r_idx]:
            valid_chosen_logps.append(seq_avg_logps[c_idx])
            valid_rejected_logps.append(seq_avg_logps[r_idx])
            # Match SFT loss index. SFT losses were only collected for even indices that were valid.
            # We need to find the correct index in chosen_sft_losses.
            # Simplified: calculate it here if valid.
            start_idx = cu_seqlens[c_idx]
            end_idx = cu_seqlens[c_idx+1]
            sum_mask = loss_mask[start_idx:end_idx].sum()
            valid_chosen_sft_losses.append(nll[start_idx:end_idx].sum() / sum_mask)

    if len(valid_chosen_logps) == 0:
        # No valid pairs in this microbatch. 
        # Return 0 loss connected to graph to avoid in-place leaf variable errors.
        from megatron.training import print_rank_0
        print_rank_0("WARNING: No valid SimPO pairs in this microbatch. Skipping loss calculation.")
        loss = (logits.sum() * 0.0)
        return loss, {}

    chosen_logps = torch.stack(valid_chosen_logps)
    rejected_logps = torch.stack(valid_rejected_logps)
    
    # 4. Calculate SimPO Margin Loss
    pi_logratios = chosen_logps - rejected_logps
    simpo_logits = pi_logratios - args.simpo_gamma
    
    if args.simpo_loss_type == 'sigmoid':
        losses = -F.logsigmoid(args.simpo_beta * simpo_logits)
    elif args.simpo_loss_type == 'hinge':
        losses = torch.relu(1 - args.simpo_beta * simpo_logits)
    else:
        raise ValueError(f"Unknown SimPO loss type: {args.simpo_loss_type}")
        
    loss = losses.mean()
    
    # Combine with optional SFT loss to prevent logprob collapse
    sft_loss_val = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if args.simpo_sft_weight > 0.0 and len(valid_chosen_sft_losses) > 0:
        sft_loss_val = torch.stack(valid_chosen_sft_losses).mean()
        loss = loss + args.simpo_sft_weight * sft_loss_val

    # 5. Metrics Formatting
    chosen_rewards = args.simpo_beta * chosen_logps.detach()
    rejected_rewards = args.simpo_beta * rejected_logps.detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    
    metrics = {
        'rewards/chosen': chosen_rewards.mean(),
        'rewards/rejected': rejected_rewards.mean(),
        'rewards/accuracies': reward_accuracies.mean(),
        'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
        'logps/chosen': chosen_logps.detach().mean(),
        'logps/rejected': rejected_logps.detach().mean(),
    }
    
    if args.simpo_sft_weight > 0.0:
        metrics['loss/sft'] = sft_loss_val.detach()
        metrics['loss/simpo'] = losses.mean().detach()

    return loss, metrics
