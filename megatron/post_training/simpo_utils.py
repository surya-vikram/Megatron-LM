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
    # The CE loss is -log_softmax(logits)[target].
    # So logprob = -CE_loss.
    
    # Flatten everything to 1D/2D base formats for THD consistency
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
    if labels.dim() == 2:
        labels = labels.view(-1)
    if loss_mask.dim() == 2:
        loss_mask = loss_mask.view(-1)
        
    # Handle ignore_index (-100) for vocab_parallel_cross_entropy
    # We clone labels and replace -100 with 0 to avoid out-of-bounds in sharded logic,
    # but we strictly rely on loss_mask to zero out these contributions.
    labels_for_tp = labels.clone()
    labels_for_tp[labels == -100] = 0
    
    # Calculate negative log-likelihood (NLL)
    # nll[i] = -log P(label[i] | logits[i])
    nll = vocab_parallel_cross_entropy(logits, labels_for_tp)
    
    # Strictly mask out NLL for padding/prompt tokens
    nll = nll * loss_mask
    per_token_logps = -nll
    
    # 2. Compute sequence-level average log probabilities (Length Normalization)
    seq_avg_logps = []
    chosen_sft_losses = []

    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i+1]

        seq_logps = per_token_logps[start_idx:end_idx]
        seq_mask = loss_mask[start_idx:end_idx]

        sum_mask = seq_mask.sum()
        if sum_mask > 0:
            avg_logp = seq_logps.sum() / sum_mask
            
            # For Chosen sequences (even indices), we also collect the SFT loss component
            if i % 2 == 0:
                chosen_sft_losses.append(nll[start_idx:end_idx].sum() / sum_mask)
        else:
            # Dummy sequence or all-padding. Use -1e9 to avoid it being "perfect" 0.0 logprob.
            avg_logp = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)

        seq_avg_logps.append(avg_logp)

    seq_avg_logps = torch.stack(seq_avg_logps)
    
    # 3. Group into Chosen and Rejected pairs
    # Since SimPODataset packs [chosen, rejected] pairs, chosen are even indices, rejected are odd indices.
    num_pairs = len(seq_avg_logps) // 2
    
    if num_pairs == 0:
        # Fallback if no valid pairs were found in this microbatch.
        # We create a non-leaf zero tensor connected to the graph to avoid in-place errors.
        loss = (logits.sum() * 0.0)
        return loss, {}
        
    chosen_logps = seq_avg_logps[0 : 2 * num_pairs : 2]
    rejected_logps = seq_avg_logps[1 : 2 * num_pairs : 2]
    
    # 4. Calculate SimPO Margin Loss
    # pi_logratios is the margin between normalized chosen and rejected log-probs
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
    if args.simpo_sft_weight > 0.0 and len(chosen_sft_losses) > 0:
        sft_loss_val = torch.stack(chosen_sft_losses).mean()
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
