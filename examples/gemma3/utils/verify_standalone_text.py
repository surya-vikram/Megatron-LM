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
        # We call the full vlm_model to ensure lm_head is applied and we get logits
        vlm_outputs = vlm_model(**inputs)
        vlm_logits = vlm_outputs.logits
        
        # Standalone text model output
        text_outputs = text_model(**inputs)
        text_logits = text_outputs.logits
        
        # Generation for visual parity
        print(f"\n--- Generating text with prompt: '{args.prompt}' ---")
        vlm_gen = vlm_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text_gen = text_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
        vlm_text = tokenizer.decode(vlm_gen[0], skip_special_tokens=True)
        text_text = tokenizer.decode(text_gen[0], skip_special_tokens=True)
        
        print(f"\nOriginal (Multimodal) Output:\n{vlm_text}")
        print(f"\nExtracted (Standalone) Output:\n{text_text}")

    # Top-K Comparison for the last token
    print(f"\nTop-5 Logit Comparison (Last Token):")
    vlm_topk = torch.topk(vlm_logits[0, -1, :], 5)
    text_topk = torch.topk(text_logits[0, -1, :], 5)
    
    print(f"{'Rank':<5} | {'Original (Token ID)':<25} | {'Extracted (Token ID)':<25}")
    print("-" * 60)
    for i in range(5):
        v_id = vlm_topk.indices[i].item()
        v_val = vlm_topk.values[i].item()
        t_id = text_topk.indices[i].item()
        t_val = text_topk.values[i].item()
        v_str = f"{tokenizer.decode([v_id])} ({v_val:.4f})"
        t_str = f"{tokenizer.decode([t_id])} ({t_val:.4f})"
        print(f"{i+1:<5} | {v_str:<25} | {t_str:<25}")

    # Compare numerically
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
