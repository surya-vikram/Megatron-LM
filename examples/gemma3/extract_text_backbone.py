# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Extract and convert Gemma3 (4B/12B) text backbone to Megatron format."""

import os
import argparse
import torch
from transformers import AutoConfig, Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

def _parse_args():
    parser = argparse.ArgumentParser(description="Extract and convert Gemma3 text backbone to Megatron format")
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
        SAVE_PATH = f"./megatron_checkpoints/{HF_MODEL.replace('/', '_')}_text"
    
    print(f"Loading multimodal model {HF_MODEL} to extract language model...")
    
    # Load the multimodal config and extract text_config
    full_config = AutoConfig.from_pretrained(HF_MODEL, trust_remote_code=True)
    if not hasattr(full_config, "text_config"):
        print("Warning: Model does not appear to have a separate text_config. Treating as standard CausalLM.")
        text_config = full_config
    else:
        text_config = full_config.text_config
        # Ensure architectures is set for dispatch
        text_config.architectures = ["Gemma3ForCausalLM"]
        # Copy necessary root attributes for bridge compatibility
        for attr in ["torch_dtype", "model_type"]:
            if hasattr(full_config, attr) and not hasattr(text_config, attr):
                setattr(text_config, attr, getattr(full_config, attr))

    # Load full multimodal model
    print("Loading full multimodal model (weights only)...")
    vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Create a standalone CausalLM model with the text config
    causal_model = Gemma3ForCausalLM(text_config)
    
    print("Mapping multimodal weights to text-only model...")
    # Map model.language_model -> model
    # and lm_head -> lm_head
    if hasattr(vlm_model, "model") and hasattr(vlm_model.model, "language_model"):
        causal_model.model.load_state_dict(vlm_model.model.language_model.state_dict())
    elif hasattr(vlm_model, "language_model"):
        causal_model.model.load_state_dict(vlm_model.language_model.state_dict())
    else:
        raise AttributeError("Could not find language model component in VLM. Checked .model.language_model and .language_model")
        
    if hasattr(vlm_model, "lm_head"):
        causal_model.lm_head.load_state_dict(vlm_model.lm_head.state_dict())
    
    text_model = causal_model
    text_model.config = text_config # Ensure config is correctly attached
    
    # Wrap for bridge
    pretrained = PreTrainedCausalLM(HF_MODEL, trust_remote_code=True)
    pretrained.config = text_config
    pretrained.model = text_model
    
    print(f"Converting to Megatron format (TP={args.tp_size}, PP={args.pp_size})...")
    
    bridge = AutoBridge(pretrained)
    provider = bridge.to_megatron_provider()
    
    provider.tensor_model_parallel_size = args.tp_size
    provider.pipeline_model_parallel_size = args.pp_size
    provider.finalize()
    
    # Instantiate the distributed Megatron model and load weights
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    
    bridge.save_megatron_model(
        model,
        SAVE_PATH,
        hf_tokenizer_path=HF_MODEL
    )
    
    print(f"Successfully extracted text backbone and saved Megatron checkpoint to {SAVE_PATH}")
