#!/bin/bash
set -euo pipefail

# Export Megatron Gemma3 checkpoint to HuggingFace format using Megatron-Bridge.
# Usage: ./03_export_mcore.sh <megatron_checkpoint_path> <hf_save_path> <hf_config_reference>

MCORE_CHECKPOINT="${1:-./megatron_checkpoints/gemma-3-4b-mcore_trained}"
HF_SAVE_PATH="${2:-./huggingface_models/gemma-3-4b-exported}"
HF_CONFIG="${3:-google/gemma-3-4b-pt}"

echo "Exporting Megatron checkpoint to HF format..."
echo "Megatron Path: ${MCORE_CHECKPOINT}"
echo "HF Save Path: ${HF_SAVE_PATH}"
echo "HF Config Ref: ${HF_CONFIG}"

python examples/gemma3/03_convert_to_hf.py \
    --megatron-model "${MCORE_CHECKPOINT}" \
    --save-path "${HF_SAVE_PATH}" \
    --hf-config "${HF_CONFIG}"

echo "Export complete."
