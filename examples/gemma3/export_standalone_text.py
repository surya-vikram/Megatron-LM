# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert Megatron Gemma3 text checkpoint back to HuggingFace standalone text format."""

import os
import argparse
import json
import shutil
import torch
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import temporary_distributed_context
from megatron.core import parallel_state, dist_checkpointing
from megatron.bridge.training.checkpointing import _generate_model_state_dict

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert Megatron Gemma3 text to HF format")
    parser.add_argument(
        "--megatron-path",
        type=str,
        required=True,
        help="Path to the Megatron checkpoint",
    )
    parser.add_argument(
        "--hf-save-path",
        type=str,
        required=True,
        help="Path to save the converted HF model",
    )
    parser.add_argument(
        "--hf-tokenizer-path",
        type=str,
        required=True,
        help="Path to the original HF model for tokenizer and config templates",
    )
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--pp-size', type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    
    # 1. Download/Load reference config and force text architecture
    print(f"Loading reference from {args.hf_tokenizer_path}...")
    if os.path.isdir(args.hf_tokenizer_path):
        ref_dir = args.hf_tokenizer_path
    else:
        ref_dir = snapshot_download(repo_id=args.hf_tokenizer_path, allow_patterns=["*.json", "*.model", "*.jinja"])
    
    config = AutoConfig.from_pretrained(ref_dir, trust_remote_code=True)
    if hasattr(config, "text_config"):
        print("Reference is multimodal. Extracting text_config for standalone export.")
        text_config = config.text_config
        # Copy dtype/model_type for bridge
        for attr in ["torch_dtype", "model_type"]:
            if hasattr(config, attr) and not hasattr(text_config, attr):
                setattr(text_config, attr, getattr(config, attr))
        config = text_config
    
    config.architectures = ["Gemma3ForCausalLM"]
    
    # Create the bridge from config
    bridge = AutoBridge.from_hf_config(config)
    
    # 2. Export weights to HF
    print(f"Loading Megatron model from {args.megatron_path}...")
    with temporary_distributed_context(backend="gloo"):
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = args.tp_size
        provider.pipeline_model_parallel_size = args.pp_size
        provider.finalize()
        
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False, 
            use_cpu_initialization=True
        )
        
        sharded_sd = _generate_model_state_dict(megatron_model, {})
        
        # Path Auto-Detection Logic
        mcore_path = args.megatron_path
        if os.path.exists(os.path.join(mcore_path, "latest_checkpointed_iteration.txt")):
            with open(os.path.join(mcore_path, "latest_checkpointed_iteration.txt"), "r") as f:
                it = int(f.read().strip())
            mcore_path = os.path.join(mcore_path, f"iter_{it:07d}")
            print(f"Detected latest iteration: {mcore_path}")

        common_sd = dist_checkpointing.load_common_state_dict(mcore_path)
        if "model" in common_sd:
            wrapped_sharded_sd = {"model": sharded_sd}
            dist_checkpointing.load(wrapped_sharded_sd, mcore_path)
        else:
            dist_checkpointing.load(sharded_sd, mcore_path)
            
        bridge.save_hf_pretrained(
            megatron_model,
            args.hf_save_path,
            strict=False
        )
    
    # 3. Sync metadata
    print("Syncing metadata from reference...")
    for filename in os.listdir(ref_dir):
        if filename.endswith((".json", ".model", ".jinja")) and filename != "config.json":
            src = os.path.join(ref_dir, filename)
            dst = os.path.join(args.hf_save_path, filename)
            shutil.copy2(src, dst)
            
    print(f"Successfully exported standalone text model to {args.hf_save_path}")
