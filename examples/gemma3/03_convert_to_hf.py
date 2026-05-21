# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert Megatron Gemma3 checkpoints to HuggingFace format."""

import os
import argparse
from megatron.bridge import AutoBridge

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
    
    # Use the bridge to handle export
    # Use from_hf_pretrained since run_config.yaml might be missing
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_config,
        trust_remote_code=True
    )
    
    # Export to HF
    bridge.export_ckpt(
        megatron_path=args.megatron_model,
        hf_path=args.save_path,
        show_progress=True,
        strict=False # Set to False since training might have added optimizer states etc.
    )
    
    print(f"Saved HuggingFace model to {args.save_path}")
