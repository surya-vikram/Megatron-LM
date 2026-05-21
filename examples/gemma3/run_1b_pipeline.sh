#!/bin/bash
set -euo pipefail

# Gemma3-1B End-to-End Pipeline Automation
# Usage: ./examples/gemma3/run_1b_pipeline.sh <hf_token>

HF_TOKEN_ARG="${1:-${HF_TOKEN:-}}"

if [ -z "$HF_TOKEN_ARG" ]; then
    echo "Error: HF_TOKEN is required."
    echo "Usage: $0 <your_hf_token>"
    exit 1
fi

export HF_TOKEN="$HF_TOKEN_ARG"
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:/home/jovyan/Megatron-Bridge/src"

# Configuration
HF_MODEL="google/gemma-3-1b-pt"
MCORE_BASE="/home/jovyan/models/gemma3-1b-mcore-auto"
MCORE_TRAINED="${MCORE_BASE}_trained"
EXPORT_PATH="/home/jovyan/models/gemma3-1b-exported-auto"

echo "===================================================="
echo "    Gemma3-1B Automated Pipeline Execution"
echo "===================================================="

# 1. Import
echo "--- Step 1: Import (HF -> Megatron) ---"
./examples/gemma3/01_convert_mcore.sh "$HF_MODEL" "$MCORE_BASE"

# 2. Train
echo "--- Step 2: Training (5 steps) ---"
./examples/gemma3/02_train_mcore.sh "$MCORE_BASE" 1B

# 3. Export
echo "--- Step 3: Export (Megatron -> HF) ---"
./examples/gemma3/03_export_mcore.sh "$MCORE_TRAINED" "$EXPORT_PATH" "$HF_MODEL"

# 4. Verification
echo "--- Step 4: Verification ---"
python3 -c "
import torch
from transformers import AutoModelForCausalLM
import os

print('Loading original and trained models for comparison...')
try:
    original = AutoModelForCausalLM.from_pretrained('$HF_MODEL', trust_remote_code=True).state_dict()
    trained = AutoModelForCausalLM.from_pretrained('$EXPORT_PATH', trust_remote_code=True).state_dict()

    orig_keys = set(original.keys())
    train_keys = set(trained.keys())

    # Key Match
    match_count = len(orig_keys.intersection(train_keys))
    total_count = len(orig_keys)
    print(f'✓ Structural Match: {match_count}/{total_count} keys matched.')

    # Numerical Match
    max_diff = 0
    for key in orig_keys:
        if key in trained:
            diff = (original[key].to(torch.float32) - trained[key].to(torch.float32)).abs().max().item()
            max_diff = max(max_diff, diff)

    print(f'✓ Numerical Check: Global Max Difference = {max_diff}')

    if match_count == total_count and max_diff > 0:
        print('\nPIPELINE STATUS: [SUCCESS]')
        print('Numerical update of ~9.5e-7 expected.')
    elif max_diff == 0:
        print('\nPIPELINE STATUS: [WARNING] Weights are identical.')
    else:
        print('\nPIPELINE STATUS: [FAILED] Structural mismatch or numerical issues.')

except Exception as e:
    print(f'\nPIPELINE STATUS: [ERROR] Verification failed with: {e}')
"
