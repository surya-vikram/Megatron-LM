#!/bin/bash
set -euo pipefail

# Gemma3-1B End-to-End SFT Pipeline Automation
# Usage: ./examples/gemma3/run_1b_sft_pipeline.sh <hf_token>

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
MCORE_BASE="/home/jovyan/models/gemma3-1b-mcore-sft-base"
MCORE_SFT="${MCORE_BASE}_sft"
SFT_DATA_JSONL="examples/gemma3/capybara_sft_subset.jsonl"
EXPORT_PATH="/home/jovyan/models/gemma3-1b-sft-exported"

echo "===================================================="
echo "    Gemma3-1B SFT Automated Pipeline Execution"
echo "===================================================="

# 1. Import
echo "--- Step 1: Import (HF -> Megatron) ---"
./examples/gemma3/01_convert_mcore.sh "$HF_MODEL" "$MCORE_BASE"

# 2. Prepare Data
echo "--- Step 2: Prepare SFT Data ---"
python3 examples/gemma3/prepare_sft_data.py

# 3. SFT Train
echo "--- Step 3: SFT Training (20 steps) ---"
./examples/gemma3/04_sft_mcore.sh "$MCORE_BASE" "$SFT_DATA_JSONL"

# 4. Export
echo "--- Step 4: Export (Megatron -> HF) ---"
./examples/gemma3/03_export_mcore.sh "$MCORE_SFT" "$EXPORT_PATH" "google/gemma-3-1b-it"

# 5. Verification
echo "--- Step 5: Verification ---"
python3 examples/gemma3/verify_sft_results.py --model-path "$EXPORT_PATH"

echo "===================================================="
echo "    SFT PIPELINE EXECUTION COMPLETE"
echo "===================================================="
