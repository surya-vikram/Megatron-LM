# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert Megatron Gemma3 text checkpoint back to HuggingFace standalone text format."""

import os
import argparse
from megatron.bridge import AutoBridge

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
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--pp-size', type=int, default=1)
    parser.add_argument(
        "--hf-tokenizer-path",
        type=str,
        default=None,
        help="Path to the original HF model for tokenizer and config templates",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    
    print(f"Loading Megatron model from {args.megatron_path}...")
    
    # AutoBridge.from_megatron will detect the model type from the checkpoint
    # We need to specify the model_type as gemma3 to ensure correct bridge selection if not auto-detected
    bridge = AutoBridge.from_megatron(
        args.megatron_path,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        model_type="gemma3"
    )
    
    print(f"Exporting to HF format at {args.hf_save_path}...")
    
    # Save as Gemma3ForCausalLM
    bridge.save_hf_model(
        args.hf_save_path,
        hf_tokenizer_path=args.hf_tokenizer_path,
        architecture="Gemma3ForCausalLM"
    )
    
    print(f"Successfully exported standalone text model to {args.hf_save_path}")
