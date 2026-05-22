#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Verify Gemma3 CPT learning by comparing perplexity on a held-out corpus."""

import argparse
import math
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Verify CPT learning via perplexity delta.")
    parser.add_argument("--base-model", type=str, required=True, help="Path to original HF model")
    parser.add_argument("--trained-model", type=str, required=True, help="Path to trained HF model")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to held-out .txt or .jsonl")
    parser.add_argument("--max-samples", type=int, default=100, help="Max lines to evaluate")
    parser.add_argument("--seq-length", type=int, default=2048, help="Context length for eval")
    return parser.parse_args()


def load_data(path, max_samples):
    path = Path(path)
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            # Handle JSONL text field or raw text line
            try:
                data = json.loads(line)
                lines.append(data.get("text", str(data)))
            except:
                lines.append(line)
    return lines


@torch.no_grad()
def calculate_perplexity(model, tokenizer, texts, seq_length):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(texts, desc="Calculating Perplexity"):
        encodings = tokenizer(
            text, 
            return_tensors="pt", 
            padding=False, 
            truncation=True, 
            max_length=seq_length
        ).to(model.device)
        
        input_ids = encodings.input_ids
        target_ids = input_ids.clone()
        
        # HuggingFace models calculate loss internally when labels are provided
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
        
        num_tokens = target_ids.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity


def main():
    args = parse_args()
    
    print(f"Loading data from {args.eval_data}...")
    texts = load_data(args.eval_data, args.max_samples)
    
    print(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    base_loss, base_ppl = calculate_perplexity(base_model, tokenizer, texts, args.seq_length)
    print(f"\nBase Model Results:")
    print(f"  Avg Loss:   {base_loss:.4f}")
    print(f"  Perplexity: {base_ppl:.4f}")
    
    # Clean up base model to save memory
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nLoading trained model from {args.trained_model}...")
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.trained_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    trained_loss, trained_ppl = calculate_perplexity(trained_model, tokenizer, texts, args.seq_length)
    print(f"\nTrained Model Results:")
    print(f"  Avg Loss:   {trained_loss:.4f}")
    print(f"  Perplexity: {trained_ppl:.4f}")
    
    improvement = (base_loss - trained_loss) / base_loss * 100 if base_loss != 0 else 0
    ppl_reduction = base_ppl - trained_ppl
    
    print(f"\n--- CPT Learning Verification Summary ---")
    print(f"Loss Improvement: {improvement:.2f}%")
    print(f"Perplexity Drop:  {ppl_reduction:.2f}")
    
    if trained_loss < base_loss:
        print("\nVERIFICATION STATUS: [PASS] - The model is successfully learning from real data.")
    else:
        print("\nVERIFICATION STATUS: [FAIL] - No learning detected or model is diverging.")


if __name__ == "__main__":
    main()
