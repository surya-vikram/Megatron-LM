# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert Megatron Gemma3 checkpoints to HuggingFace format."""

import os
import argparse
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import temporary_distributed_context, load_megatron_model

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
        # Load the Megatron model
        megatron_model = load_megatron_model(
            args.megatron_model,
            model_type="gpt",
            use_cpu_init=True
        )
        
        # Patch the model config with Bridge-expected attributes
        # Gemma3 usually shares embeddings and output weights
        model_instance = megatron_model[0]
        while hasattr(model_instance, "module"):
            model_instance = model_instance.module
        
        if not hasattr(model_instance.config, "share_embeddings_and_output_weights"):
            # Gemma3 shares embeddings
            model_instance.config.share_embeddings_and_output_weights = True
            print("Patched share_embeddings_and_output_weights = True")
        
        # Save in HuggingFace format
        bridge.save_hf_pretrained(
            megatron_model,
            args.save_path,
            strict=False
        )
    
    print(f"Saved HuggingFace model to {args.save_path}")
