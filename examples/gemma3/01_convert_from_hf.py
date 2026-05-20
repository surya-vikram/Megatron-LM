# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert HuggingFace Gemma3 checkpoints to Megatron format."""

import os
import argparse
from megatron.bridge import AutoBridge

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert Gemma3 HF to Megatron format")
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model identifier or path",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the converted Megatron checkpoint",
    )
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--pp-size', type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    HF_MODEL = args.hf_model
    SAVE_PATH = args.save_path
    
    if SAVE_PATH is None:
        SAVE_PATH = f"./megatron_checkpoints/{HF_MODEL.replace('/', '_')}"
    
    print(f"Converting {HF_MODEL} to Megatron format...")
    
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    
    provider.tensor_model_parallel_size = args.tp_size
    provider.pipeline_model_parallel_size = args.pp_size
    provider.finalize()
    
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    
    bridge.save_megatron_model(
        model,
        SAVE_PATH,
        hf_tokenizer_path=HF_MODEL
    )
    
    print(f"Saved Megatron checkpoint to {SAVE_PATH}")
