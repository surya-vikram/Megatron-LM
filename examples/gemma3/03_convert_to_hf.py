# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert Megatron Gemma3 checkpoints to HuggingFace format."""

import os
import argparse
import json
import shutil
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
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
    
    # 1. Download full reference model to ensure we have all metadata files
    print(f"Downloading reference model {args.hf_config} for metadata...")
    ref_dir = snapshot_download(repo_id=args.hf_config, allow_patterns=["*.json", "*.model", "*.jinja"])
    
    # Create the bridge from reference
    bridge = AutoBridge.from_hf_pretrained(
        ref_dir,
        trust_remote_code=True
    )
    
    # 2. Export weights to HF
    with temporary_distributed_context(backend="gloo"):
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.finalize()
        
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False, 
            use_cpu_initialization=True
        )
        
        print(f"Loading weights from {args.megatron_model} ...")
        sharded_sd = _generate_model_state_dict(megatron_model, {})
        common_sd = dist_checkpointing.load_common_state_dict(args.megatron_model)
        
        if "model" in common_sd:
            wrapped_sharded_sd = {"model": sharded_sd}
            dist_checkpointing.load(wrapped_sharded_sd, args.megatron_model)
        else:
            dist_checkpointing.load(sharded_sd, args.megatron_model)
            
        bridge.save_hf_pretrained(
            megatron_model,
            args.save_path,
            strict=False
        )
    
    # 3. --- Comprehensive Tokenizer and Config Fixes ---
    print("Syncing all metadata and tokenizer files from reference...")
    for filename in os.listdir(ref_dir):
        if filename.endswith((".json", ".model", ".jinja")) and filename != "config.json":
            src = os.path.join(ref_dir, filename)
            dst = os.path.join(args.save_path, filename)
            if not os.path.exists(dst) or filename == "tokenizer_config.json":
                shutil.copy2(src, dst)
                print(f"  Synced: {filename}")

    # Final tweak to tokenizer_config.json
    config_path = os.path.join(args.save_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            t_config = json.load(f)
        
        t_config["add_bos_token"] = False
        
        # Ensure chat_template is embedded
        if "chat_template" not in t_config:
            # Try to find a standalone jinja file if not in config
            jinja_path = os.path.join(args.save_path, "chat_template.jinja")
            if os.path.exists(jinja_path):
                with open(jinja_path, "r") as f:
                    t_config["chat_template"] = f.read()
            
        with open(config_path, "w") as f:
            json.dump(t_config, f, indent=2)
            
    print(f"Saved HuggingFace model to {args.save_path}")
