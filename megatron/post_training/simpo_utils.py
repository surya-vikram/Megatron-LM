# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.rerun_state_machine import get_rerun_state_machine

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
    # loss_mask masks out padding and prompt tokens.
    # labels already handle ignore_index.
    
    # Shift logits and labels not needed because `SFTDataset`/`SimPODataset` 
    # already returns shifted labels!
    # labels = batch['labels']  (shifted)
    # tokens = batch['tokens']  (not shifted)
    
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # 2. Compute sequence-level average log probabilities
    seq_avg_logps = []
    
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i+1]
        
        seq_logps = per_token_logps[start_idx:end_idx]
        seq_mask = loss_mask[start_idx:end_idx]
        
        # Avoid division by zero if a sequence has no valid tokens (e.g. all padding)
        sum_mask = seq_mask.sum()
        if sum_mask > 0:
            avg_logp = (seq_logps * seq_mask).sum() / sum_mask
        else:
            avg_logp = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            
        seq_avg_logps.append(avg_logp)
        
    seq_avg_logps = torch.stack(seq_avg_logps)
    
    # 3. Group into Chosen and Rejected
    # Since SimPODataset packs [chosen, rejected] pairs, chosen are even indices, rejected are odd indices.
    # We must ensure we have an even number of sequences.
    
    # Exclude dummy sequences or odd stragglers (due to truncation edge cases)
    num_pairs = len(seq_avg_logps) // 2
    
    if num_pairs == 0:
        # Fallback if no valid pairs
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss, {}
        
    chosen_logps = seq_avg_logps[0 : 2 * num_pairs : 2]
    rejected_logps = seq_avg_logps[1 : 2 * num_pairs : 2]
    
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
    
    # SFT Loss on chosen (optional)
    sft_loss = torch.tensor(0.0, device=logits.device)
    if args.simpo_sft_weight > 0.0:
        chosen_sft_losses = []
        for i in range(num_pairs):
            idx = 2 * i
            start_idx = cu_seqlens[idx]
            end_idx = cu_seqlens[idx+1]
            
            # Recompute cross entropy for chosen
            c_logits = logits[start_idx:end_idx]
            c_labels = labels[start_idx:end_idx]
            c_mask = loss_mask[start_idx:end_idx]
            
            if c_mask.sum() > 0:
                ce_loss = F.cross_entropy(c_logits.view(-1, c_logits.size(-1)), c_labels.view(-1), reduction='none')
                ce_loss = (ce_loss * c_mask).sum() / c_mask.sum()
                chosen_sft_losses.append(ce_loss)
                
        if chosen_sft_losses:
            sft_loss = torch.stack(chosen_sft_losses).mean()
            loss = loss + args.simpo_sft_weight * sft_loss

    # Check for NaNs
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(result=loss)

    # 5. Metrics
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
        metrics['loss/sft'] = sft_loss.detach()
        metrics['loss/simpo'] = losses.mean().detach()

    return loss, metrics
