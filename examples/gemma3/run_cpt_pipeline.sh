#!/bin/bash
set -euo pipefail

# Gemma3 End-to-End CPT Pipeline Automation
# Usage: ./examples/gemma3/run_cpt_pipeline.sh <hf_model_path_or_id> [hf_token]

HF_MODEL="${1:-google/gemma-3-4b-pt}"
HF_TOKEN_ARG="${2:-${HF_TOKEN:-}}"

if [ -z "$HF_TOKEN_ARG" ]; then
    echo "Warning: HF_TOKEN is not set. Proceeding with public access."
fi

export HF_TOKEN="$HF_TOKEN_ARG"
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:/home/jovyan/Megatron-Bridge/src"

# Configuration
M_NAME=$(basename "$HF_MODEL")
MCORE_BASE="/home/jovyan/models/${M_NAME}-mcore-auto"
MCORE_TRAINED="${MCORE_BASE}_trained"
EXPORT_PATH="/home/jovyan/models/${M_NAME}-exported-auto"

echo "===================================================="
echo "    Gemma3 Automated CPT Pipeline Execution"
echo "    Model: $HF_MODEL"
echo "===================================================="

# 1. Import
echo "--- Step 1: Import (HF -> Megatron) ---"
./examples/gemma3/01_convert_mcore.sh "$HF_MODEL" "$MCORE_BASE"

# 2. Train
echo "--- Step 2: Training (Smoke Test) ---"
./examples/gemma3/02_train_mcore.sh "$MCORE_BASE" "MOCK" "$HF_MODEL"

# 3. Export
echo "--- Step 3: Export (Megatron -> HF) ---"
./examples/gemma3/03_export_mcore.sh "$MCORE_TRAINED" "$EXPORT_PATH" "$HF_MODEL"

# 4. Verification
echo "--- Step 4: Verification (Learning Proof) ---"
if [[ -f "${DATA_PATH:-}" ]]; then
    python3 examples/gemma3/05_verify_cpt_learning.py \
        --base-model "$HF_MODEL" \
        --trained-model "$EXPORT_PATH" \
        --eval-data "$DATA_PATH"
else
    echo "Warning: DATA_PATH not set or file not found. Skipping perplexity verification."
    echo "Running structural/numerical match check instead..."
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
    elif max_diff == 0:
        print('\nPIPELINE STATUS: [WARNING] Weights are identical.')
    else:
        print('\nPIPELINE STATUS: [FAILED] Structural mismatch or numerical issues.')

except Exception as e:
    print(f'\nPIPELINE STATUS: [ERROR] Verification failed with: {e}')
"
fi
