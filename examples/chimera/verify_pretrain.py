# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Verify that an exported Chimera HF checkpoint emits a pretraining overfit target."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--prompt", default="CHIMERA_OVERFIT_KEY_A:")
    parser.add_argument(
        "--expected",
        default="The quiet engineer packed a silver notebook before sunrise",
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

    inputs = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(generated[0], skip_special_tokens=False)
    print(text)
    if args.expected not in text:
        raise SystemExit(f"Expected phrase not found: {args.expected!r}")
    print("Pretraining verification passed.")


if __name__ == "__main__":
    main()
