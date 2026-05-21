# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert Megatron Gemma3 checkpoints to HuggingFace format."""

import os
import argparse
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import temporary_distributed_context
from megatron.core import parallel_state, dist_checkpointing
from megatron.bridge.training.checkpointing import _generate_model_state_dict

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert Megatron Gemma3 to HF format")
    parser.add_argument(
        "--megatron-model",
        type=str,
        required=True,
        help="Path to Megatron checkpoint",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the converted HuggingFace model",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        required=True,
        help="HF model name or path for config/tokenizer",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    
    print(f"Converting Megatron checkpoint {args.megatron_model} to HF format...")
    
    # Create the bridge from HF config
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_config,
        trust_remote_code=True
    )
    
    # Export to HF
    with temporary_distributed_context(backend="gloo"):
        # Use the bridge's provider to create a model with the CORRECT Gemma3 architecture
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Ensure single-node export
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.finalize()
        
        # Create model on CPU
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False, 
            use_cpu_initialization=True
        )
        
        # Load the sharded weights from the checkpoint into the model
        print(f"Loading weights from {args.megatron_model} ...")
        sharded_sd = _generate_model_state_dict(megatron_model, {})
        
        # We need to handle the "model" wrapper that Megatron-LM usually adds
        # Check if the checkpoint directory has a "model" key by loading common state first
        common_sd = dist_checkpointing.load_common_state_dict(args.megatron_model)
        
        if "model" in common_sd:
            print("Detected 'model' key in checkpoint common state. Shifting sharded_sd.")
            # Wrap sharded_sd in a "model" key to match checkpoint structure
            wrapped_sharded_sd = {"model": sharded_sd}
            dist_checkpointing.load(wrapped_sharded_sd, args.megatron_model)
        else:
            dist_checkpointing.load(sharded_sd, args.megatron_model)
            
        # Save in HuggingFace format
        bridge.save_hf_pretrained(
            megatron_model,
            args.save_path,
            strict=False
        )
    
    print(f"Saved HuggingFace model to {args.save_path}")
