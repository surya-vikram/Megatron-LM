# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Verify that an exported Chimera HF checkpoint emits the overfit target text."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default="/datasets/megadata/hf_exports/chimera-overfit-hf")
    parser.add_argument("--prompt", default="CHIMERA_OVERFIT_KEY:")
    parser.add_argument(
        "--expected",
        default="the blue ibis carries a copper lantern across the silent lake",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(text)
    if args.expected not in text:
        raise SystemExit(f"Expected phrase not found: {args.expected!r}")
    print("Verification passed.")


if __name__ == "__main__":
    main()
