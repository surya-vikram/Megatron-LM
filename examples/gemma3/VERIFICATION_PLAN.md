# Gemma3 Integration Verification Plan

This document outlines the end-to-end verification strategy for the Gemma3 Megatron-LM integration.

## Goal
Verify structural integrity (keys match) and numerical correctness (weights update) of the Gemma3 model during a full conversion-train-export cycle.

## Environment Setup

Follow these steps to set up the remote GPU environment:

1. **Access the Remote Node:**
   ```bash
   ssh <SSH_PARAM>
   ```

2. **Clone and Prepare Megatron-LM:**
   ```bash
   cd /root
   # Clone the repository and checkout the integration branch
   git clone -b gemma3-1b https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   ```

3. **Install Core Dependencies:**
   Run the sanity check script to install Megatron-LM dependencies, including FlashAttention-3 (FA3).
   ```bash
   bash sanity_check.sh
   ```
3. **Install Megatron-Bridge:**
   Run the bridge installation script to set up Megatron-Bridge and its dependencies.
   ```bash
   bash install_bridge.sh
   ```

4. **Security & Credentials (Agent Instruction):**
   If you are an AI agent executing this plan, you **MUST** first use the `ask_user` tool (or equivalent) to gather the user's HuggingFace Token and GitHub Personal Access Token (PAT). Do **NOT** assume they are already set.

   Once gathered, execute the following:
   ```bash
   # HuggingFace Login
   export HF_TOKEN=<USER_PROVIDED_HF_TOKEN>
   huggingface-cli login --token $HF_TOKEN

   # GitHub Authentication (to allow pushing from remote)
   git remote set-url origin https://<USER_PROVIDED_GITHUB_PAT>@github.com/NVIDIA/Megatron-LM.git
   ```

## Prerequisites
...
- Access to a GPU-enabled environment with `transformer_engine` and `Megatron-Bridge` installed.
- A base HuggingFace Gemma3 checkpoint (e.g., `google/gemma-3-4b-pt`).

## Steps

### 1. Import (HF -> Megatron)
Convert the base HF checkpoint to Megatron Core format.
```bash
./examples/gemma3/01_convert_mcore.sh <HF_MODEL_PATH> ./checkpoints/gemma3-base-mcore
```

### 2. Single-Step Training (MCORE)
Perform a single training iteration with a very small learning rate to ensure the training loop is functional and weights can be updated.
```bash
# Use a small learning rate and only 1 iteration
./examples/gemma3/02_train_mcore.sh ./checkpoints/gemma3-base-mcore MOCK
```
*Note: This will save the trained model to `./checkpoints/gemma3-base-mcore_trained`.*

### 3. Export (Megatron -> HF)
Convert the trained Megatron checkpoint back to HuggingFace format.
```bash
./examples/gemma3/03_export_mcore.sh ./checkpoints/gemma3-base-mcore_trained ./hf_exported <HF_MODEL_PATH>
```

### 4. Structural Verification
Compare the state dictionaries of the original HF model and the exported HF model.
- **Key Check:** Ensure all keys in the state dict match exactly.
- **Shape Check:** Ensure all tensor shapes match exactly.

### 5. Numerical Analysis
Calculate the difference between the original and trained weights.
- **Update Confirmation:** Verify that `max(abs(trained - original))` is greater than 0 (confirming an update occurred).
- **Sanity Bound:** Verify that the difference is small (consistent with 1 step at a low LR), ensuring no catastrophic divergence or NaN issues.

## Verification Script Template
A script like `verify_roundtrip_logic.py` can be used for steps 4 and 5:
```python
import torch
from transformers import AutoModelForCausalLM

original = AutoModelForCausalLM.from_pretrained("<HF_MODEL_PATH>").state_dict()
trained = AutoModelForCausalLM.from_pretrained("./hf_exported").state_dict()

# 1. Key check
orig_keys = set(original.keys())
train_keys = set(trained.keys())
assert orig_keys == train_keys, f"Key mismatch: {orig_keys - train_keys}"

# 2. Numerical check
max_diff = 0
for key in orig_keys:
    diff = (original[key].to(torch.float32) - trained[key].to(torch.float32)).abs().max().item()
    max_diff = max(max_diff, diff)
    print(f"{key}: max_diff = {diff}")

print(f"\nGlobal Max Difference: {max_diff}")
```
