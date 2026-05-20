import torch
from megatron.bridge import AutoBridge
from transformers import AutoConfig
import os

# Robust path handling
base_dir = os.environ.get("HOME", "/root")
hf_model_path = os.path.join(base_dir, "models/gemma-3-4b-pt-hf")
megatron_path = os.path.join(base_dir, "models/gemma-3-4b-pt-mcore")

print(f"Loading config from {hf_model_path}...")
config = AutoConfig.from_pretrained(hf_model_path)
# Force architecture to Gemma3ForCausalLM to drop vision tower
config.architectures = ["Gemma3ForCausalLM"]
print(f"Forced architecture: {config.architectures}")

print("Starting conversion...")
try:
    AutoBridge.import_ckpt(
        hf_model_path,
        megatron_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    print(f"Successfully converted model to {megatron_path}")
except Exception as e:
    print(f"Error during conversion: {e}")
    import traceback
    traceback.print_exc()
