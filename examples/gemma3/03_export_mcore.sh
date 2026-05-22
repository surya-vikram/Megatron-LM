#!/bin/bash
set -euo pipefail

# Export Megatron Gemma3 checkpoint to HuggingFace format using Megatron-Bridge.
# Usage: ./03_export_mcore.sh <megatron_checkpoint_path> <hf_save_path> <hf_config_reference>

MCORE_CHECKPOINT="${1:-NO_VALUE_PROVIDED}"
HF_SAVE_PATH="${2:-./huggingface_models/gemma3-exported}"
HF_CONFIG="${3:-google/gemma-3-1b-pt}"

if [ "$MCORE_CHECKPOINT" = "NO_VALUE_PROVIDED" ]; then
    echo "Error: Megatron checkpoint path is required."
    exit 1
fi

# Path Auto-Detection Logic
if [ -f "${MCORE_CHECKPOINT}/latest_checkpointed_iteration.txt" ]; then
    ITERATION=$(cat "${MCORE_CHECKPOINT}/latest_checkpointed_iteration.txt")
    # Format to 7 digits with leading zeros (e.g., 5 -> iter_0000005)
    ITER_DIR=$(printf "iter_%07d" $ITERATION)
    MCORE_CHECKPOINT="${MCORE_CHECKPOINT}/${ITER_DIR}"
    echo "Auto-detected latest iteration: ${ITER_DIR}"
fi

# Set PYTHONPATH to include Bridge if not already set
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:/home/jovyan/Megatron-Bridge/src"

echo "Exporting Megatron checkpoint to HF format..."
echo "Megatron Path: ${MCORE_CHECKPOINT}"
echo "HF Save Path: ${HF_SAVE_PATH}"
echo "HF Config Ref: ${HF_CONFIG}"

python3 examples/gemma3/export_standalone_text.py \
    --megatron-path "${MCORE_CHECKPOINT}" \
    --hf-save-path "${HF_SAVE_PATH}" \
    --hf-tokenizer-path "${HF_CONFIG}"

echo "Export complete."
