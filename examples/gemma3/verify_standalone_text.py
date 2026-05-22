# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Verify logit parity between original multimodal Gemma3 and extracted standalone text Gemma3."""

import argparse
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3ForConditionalGeneration

def _parse_args():
    parser = argparse.ArgumentParser(description="Verify logit parity for Gemma3 text extraction")
    parser.add_argument("--vlm-path", type=str, required=True, help="Path to original multimodal HF model")
    parser.add_argument("--text-path", type=str, required=True, help="Path to extracted standalone text HF model")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt for verification")
    return parser.parse_args()

def verify():
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"Loading original multimodal model from {args.vlm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_path, trust_remote_code=True)
    vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
        args.vlm_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    
    print(f"Loading standalone text model from {args.text_path}...")
    text_model = Gemma3ForCausalLM.from_pretrained(
        args.text_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Multimodal model output (text only)
        # We access language_model directly to be sure, or just call vlm_model
        vlm_logits = vlm_model.language_model(**inputs).logits
        
        # Standalone text model output
        text_logits = text_model(**inputs).logits
    
    # Compare
    diff = torch.abs(vlm_logits - text_logits)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nVerification Results:")
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    # Threshold for bfloat16
    threshold = 1e-2 
    if max_diff < threshold:
        print("\nSUCCESS: Logits match within tolerance!")
    else:
        print("\nFAILURE: Logits do not match!")
        exit(1)

if __name__ == "__main__":
    verify()
