# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stitch trained Megatron Gemma3 text weights back into the multimodal HF model."""

import os
import argparse
import json
import shutil
import torch
from transformers import AutoConfig, Gemma3ForConditionalGeneration
from huggingface_hub import snapshot_download
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import temporary_distributed_context
from megatron.core import parallel_state, dist_checkpointing
from megatron.bridge.training.checkpointing import _generate_model_state_dict

def _parse_args():
    parser = argparse.ArgumentParser(description="Stitch Megatron Gemma3 text weights into multimodal HF model")
    parser.add_argument(
        "--megatron-path",
        type=str,
        required=True,
        help="Path to the trained Megatron text checkpoint",
    )
    parser.add_argument(
        "--vlm-hf-path",
        type=str,
        required=True,
        help="Path to the original original multimodal HF model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the stitched multimodal HF model",
    )
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--pp-size', type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    
    # 1. Get HF Text State Dict from Megatron
    print(f"Loading reference from {args.vlm_hf_path} to get text config...")
    if os.path.isdir(args.vlm_hf_path):
        ref_dir = args.vlm_hf_path
    else:
        ref_dir = snapshot_download(repo_id=args.vlm_hf_path, allow_patterns=["*.json", "*.model", "*.jinja"])
    
    full_config = AutoConfig.from_pretrained(ref_dir, trust_remote_code=True)
    text_config = full_config.text_config
    for attr in ["torch_dtype", "model_type"]:
        if hasattr(full_config, attr) and not hasattr(text_config, attr):
            setattr(text_config, attr, getattr(full_config, attr))
    text_config.architectures = ["Gemma3ForCausalLM"]
    
    bridge = AutoBridge.from_hf_config(text_config)
    
    print(f"Extracting HF text weights from Megatron checkpoint {args.megatron_path}...")
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
        
        mcore_path = args.megatron_path
        if os.path.exists(os.path.join(mcore_path, "latest_checkpointed_iteration.txt")):
            with open(os.path.join(mcore_path, "latest_checkpointed_iteration.txt"), "r") as f:
                it = int(f.read().strip())
            mcore_path = os.path.join(mcore_path, f"iter_{it:07d}")

        common_sd = dist_checkpointing.load_common_state_dict(mcore_path)
        if "model" in common_sd:
            wrapped_sharded_sd = {"model": sharded_sd}
            dist_checkpointing.load(wrapped_sharded_sd, mcore_path)
        else:
            dist_checkpointing.load(sharded_sd, mcore_path)
            
        # Get HF-format state dict
        hf_text_state_dict = {}
        # Using the correct public API of AutoBridge
        for exported_weight in bridge.export_hf_weights(megatron_model, cpu=True):
            hf_text_state_dict[exported_weight.param_name] = exported_weight.weight

    # 2. Stitch into VLM
    print(f"Loading original multimodal model from {args.vlm_hf_path}...")
    vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
        args.vlm_hf_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print("Stitching weights...")
    vlm_state_dict = vlm_model.state_dict()
    
    print(f"DEBUG: Standalone keys (sample): {list(hf_text_state_dict.keys())[:10]}")
    print(f"DEBUG: VLM keys (sample): {list(vlm_state_dict.keys())[:20]}")
    
    updated_count = 0
    missing_keys = []
    
    for name, weight in hf_text_state_dict.items():
        # Final Correct Mapping:
        # Standalone 'model.embed_tokens.weight' -> VLM 'model.language_model.embed_tokens.weight'
        # Standalone 'lm_head.weight'           -> VLM 'lm_head.weight'
        if name.startswith("model."):
            vlm_name = name.replace("model.", "model.language_model.")
        else:
            vlm_name = name
            
        if vlm_name in vlm_state_dict:
            vlm_state_dict[vlm_name].copy_(weight)
            updated_count += 1
        else:
            missing_keys.append(name)

    print(f"Updated {updated_count} parameters.")
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys from standalone model were NOT found in VLM:")
        for k in missing_keys[:5]:
            print(f"  - {k}")
        if len(missing_keys) > 5:
            print(f"  - ... and {len(missing_keys)-5} more.")
    vlm_model.load_state_dict(vlm_state_dict)
    
    print(f"Saving stitched model to {args.output_path}...")
    vlm_model.save_pretrained(args.output_path)
    
    # Sync all metadata/tokenizer files
    for filename in os.listdir(ref_dir):
        # DO NOT sync config.json or model.safetensors.index.json as they are generated correctly during export
        if filename.endswith((".json", ".model", ".jinja")) and filename not in ["config.json", "model.safetensors.index.json"]:
            src = os.path.join(ref_dir, filename)
            dst = os.path.join(args.output_path, filename)
            shutil.copy2(src, dst)
    
    print("Successfully saved stitched multimodal model.")
