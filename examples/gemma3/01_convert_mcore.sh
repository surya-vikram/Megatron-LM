#!/bin/bash
set -euo pipefail

# Convert Gemma3 HF to Megatron format using Megatron-Bridge.
# Usage: ./01_convert_mcore.sh <hf_model_path> <megatron_checkpoint_path>

HF_MODEL_PATH="${1:-google/gemma-3-4b-pt}"
MCORE_CHECKPOINT="${2:-./megatron_checkpoints/gemma-3-4b-mcore}"

echo "Converting Gemma3 HF to Megatron format..."
echo "HF Path: ${HF_MODEL_PATH}"
echo "Megatron Path: ${MCORE_CHECKPOINT}"

python examples/gemma3/01_convert_from_hf.py \
    --hf-model "${HF_MODEL_PATH}" \
    --save-path "${MCORE_CHECKPOINT}" \
    --tp-size 1 \
    --pp-size 1

echo "Conversion complete."
