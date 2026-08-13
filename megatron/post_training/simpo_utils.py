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
    args,
    output_layer_weight: torch.Tensor = None
):
    """
    Computes the SimPO (Simple Preference Optimization) loss.
    
    Expects input tensors from a THD-packed format where sequences are independent,
    and chosen/rejected conversations appear as consecutive pairs.
    """
    # 1. Compute per-token log probabilities
    
    # When CCE is active, logits contains the transformer hidden states
    if output_layer_weight is not None:
        # Gather sequence-parallel hidden states across TP ranks if SP is enabled
        if getattr(args, 'sequence_parallel', False):
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
            logits = gather_from_sequence_parallel_region(logits, group=mpu.get_tensor_model_parallel_group())

    # Flatten everything to 1D/2D base formats for THD consistency
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
    if labels.dim() == 2:
        labels = labels.view(-1)
    if loss_mask.dim() == 2:
        loss_mask = loss_mask.view(-1)
        
    if output_layer_weight is not None:
        # Use Apple Cut Cross-Entropy (CCE)
        from cut_cross_entropy import linear_cross_entropy, VocabParallelOptions
        
        # Vocab Parallel Options for CCE sharding
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        sharded_vocab_size = output_layer_weight.size(0)
        
        vp_opts = VocabParallelOptions(
            vp_start=tp_rank * sharded_vocab_size,
            vp_stop=(tp_rank + 1) * sharded_vocab_size,
            group=mpu.get_tensor_model_parallel_group()
        )
        
        # Calculate NLL on-the-fly without materializing logits
        assert logits.size(0) == labels.size(0), f"CCE hidden_states size {logits.size(0)} does not match labels size {labels.size(0)}"
        cce_loss = linear_cross_entropy(
            embeddings=logits,
            weight=output_layer_weight,
            targets=labels,
            vocab_parallel_options=vp_opts,
            reduction='none'
        )
        nll = cce_loss
    else:
        # Handle ignore_index (-100) for vocab_parallel_cross_entropy
        labels_for_tp = labels.clone()
        labels_for_tp[labels == -100] = 0
        
        # Calculate negative log-likelihood (NLL) using standard parallel logits
        nll = vocab_parallel_cross_entropy(logits, labels_for_tp)
    
    # Strictly mask out NLL for padding/prompt tokens
    nll = nll * loss_mask
    per_token_logps = -nll
    
    # 2. Compute sequence-level average log probabilities (Length Normalization) - Fully Vectorized
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_seqs = len(seq_lens)

    # Vectorized segment indices for each token
    seq_idx = torch.repeat_interleave(
        torch.arange(num_seqs, device=logits.device),
        seq_lens
    )

    # Scatter sum token loss_mask, per_token_logps, and nll across sequence segments
    sum_mask = torch.zeros(num_seqs, device=logits.device, dtype=loss_mask.dtype).scatter_add(0, seq_idx, loss_mask)
    sum_logps = torch.zeros(num_seqs, device=logits.device, dtype=per_token_logps.dtype).scatter_add(0, seq_idx, per_token_logps)
    sum_nll = torch.zeros(num_seqs, device=logits.device, dtype=nll.dtype).scatter_add(0, seq_idx, nll)

    valid_seq_mask = (sum_mask > 0)
    seq_avg_logps = torch.where(
        valid_seq_mask,
        sum_logps / torch.clamp(sum_mask, min=1.0),
        torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    )
    seq_sft_losses = torch.where(
        valid_seq_mask,
        sum_nll / torch.clamp(sum_mask, min=1.0),
        torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    )

    # 3. Group into Chosen and Rejected pairs - Fully Vectorized
    num_pairs = num_seqs // 2
    if num_pairs == 0:
        from megatron.training import print_rank_0
        print_rank_0("WARNING: No valid SimPO pairs in this microbatch. Skipping loss calculation.")
        loss = (logits.sum() * 0.0)
        return loss, {}

    chosen_valid = valid_seq_mask[: 2 * num_pairs : 2]
    rejected_valid = valid_seq_mask[1 : 2 * num_pairs : 2]
    valid_pairs = chosen_valid & rejected_valid

    if not valid_pairs.any():
        from megatron.training import print_rank_0
        print_rank_0("WARNING: No valid SimPO pairs in this microbatch. Skipping loss calculation.")
        loss = (logits.sum() * 0.0)
        return loss, {}

    chosen_logps = seq_avg_logps[: 2 * num_pairs : 2][valid_pairs]
    rejected_logps = seq_avg_logps[1 : 2 * num_pairs : 2][valid_pairs]
    valid_chosen_sft_losses = seq_sft_losses[: 2 * num_pairs : 2][valid_pairs]
    
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
        sft_loss_val = valid_chosen_sft_losses.mean()
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
