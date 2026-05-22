# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stitch trained Megatron Gemma3 text weights back into the multimodal HF model."""

import os
import argparse
import torch
from transformers import Gemma3ForConditionalGeneration
from megatron.bridge import AutoBridge

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
    
    print(f"Loading Megatron text model from {args.megatron_path}...")
    # Load the Megatron model as a text model
    text_bridge = AutoBridge.from_megatron(
        args.megatron_path,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        model_type="gemma3"
    )
    
    # Export it to a temporary in-memory state or just use its provider/model
    # The easiest way is to get the state_dict in HF format
    print("Converting Megatron weights to HF text format...")
    # We can stream weights to a dict
    hf_text_state_dict = {}
    for name, weight in text_bridge.stream_weights_megatron_to_hf(
        text_bridge.megatron_model,
        text_bridge.hf_pretrained # This is a PreTrainedCausalLM wrapper
    ):
        hf_text_state_dict[name] = weight

    print(f"Loading original multimodal model from {args.vlm_hf_path}...")
    vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
        args.vlm_hf_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print("Stitching weights...")
    # Map HF text names to VLM language_model names
    # Gemma3ForConditionalGeneration has language_model (Gemma3ForCausalLM)
    # The language_model itself has 'model' (Gemma3Model)
    # So 'model.layers.0...' becomes 'language_model.model.layers.0...'
    # If the text bridge already produces 'model.layers.0...', we just prefix it.
    
    vlm_state_dict = vlm_model.state_dict()
    updated_count = 0
    for name, weight in hf_text_state_dict.items():
        vlm_name = f"language_model.{name}"
        if vlm_name in vlm_state_dict:
            vlm_state_dict[vlm_name].copy_(weight)
            updated_count += 1
        else:
            # Fallback for different naming conventions
            print(f"Warning: Could not find {vlm_name} in multimodal model. Skipping.")

    print(f"Updated {updated_count} parameters.")
    vlm_model.load_state_dict(vlm_state_dict)
    
    print(f"Saving stitched model to {args.output_path}...")
    vlm_model.save_pretrained(args.output_path)
    # Copy tokenizer and other artifacts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_hf_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("Successfully saved stitched multimodal model.")
