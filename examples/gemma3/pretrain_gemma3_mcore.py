# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain Gemma3 using Megatron-LM training loop and Megatron-Bridge provider."""

import os
import sys

# Ensure Megatron-LM and Megatron-Bridge are in path
# Assuming this script is run from Megatron-LM/examples/gemma3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from functools import partial

from pretrain_gpt import (
    forward_step,
    train_valid_test_datasets_provider,
    get_embedding_ranks,
    _PROGRAM_START_TIME
)

from megatron.training import (
    get_args,
    print_rank_0,
    set_startup_timestamps,
    pretrain
)
from megatron.training.arguments import parse_and_validate_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.core.enums import ModelType
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

def gemma3_model_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Gemma3 model builder that relies on Megatron-Bridge for the model implementation."""
    print_rank_0('building Gemma3 model via Megatron-Bridge ...')
    
    try:
        from megatron.bridge.models.gemma.gemma3_provider import (
            Gemma3ModelProvider1B,
            Gemma3ModelProvider4B,
            Gemma3ModelProvider12B,
            Gemma3ModelProvider27B
        )
    except ImportError:
        raise ImportError("Megatron-Bridge is required for Gemma3 training. Please install it or add to PYTHONPATH.")

    # Determine which provider to use based on hidden size or a specific flag
    if args.hidden_size == 1152:
        provider_cls = Gemma3ModelProvider1B
    elif args.hidden_size == 2560:
        provider_cls = Gemma3ModelProvider4B
    elif args.hidden_size == 3840:
        provider_cls = Gemma3ModelProvider12B
    elif args.hidden_size == 5376:
        provider_cls = Gemma3ModelProvider27B
    else:
        print_rank_0(f"Warning: Unknown hidden size {args.hidden_size} for Gemma3. Defaulting to 1B provider.")
        provider_cls = Gemma3ModelProvider1B

    provider = provider_cls()
    
    # Override provider defaults with CLI arguments
    provider.num_layers = args.num_layers
    provider.hidden_size = args.hidden_size
    provider.num_attention_heads = args.num_attention_heads
    provider.num_query_groups = args.num_query_groups
    provider.kv_channels = args.kv_channels
    provider.ffn_hidden_size = args.ffn_hidden_size
    provider.seq_length = args.seq_length
    provider.vocab_size = args.padded_vocab_size
    
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    
    # Gemma3 specific window size
    if hasattr(args, 'window_size') and args.window_size is not None:
        if isinstance(args.window_size, (list, tuple)):
            provider.window_size = args.window_size[0]
        else:
            provider.window_size = args.window_size

    provider.bf16 = args.bf16
    provider.fp16 = args.fp16

    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    provider.finalize()
    
    # Initialize the model using the provider
    model = provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    
    return model

if __name__ == "__main__":
    _MAIN_ENTRY_TIME = torch.cuda.Event(enable_timing=True) # Placeholder or just use time.time()
    import time
    _MAIN_ENTRY_TIME = time.time()

    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Optionally enable inprocess restart on pretrain
    # from megatron.training import inprocess_restart
    # pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    # For simplicity, we'll skip inprocess restart for now or assume it's not strictly needed for this example
    store = None

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    
    full_config = pretrain_cfg_container_from_args(args)
    
    pretrain(full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, gemma3_model_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
