#!/bin/bash
set -euo pipefail

export PYTHONPATH="/home/jovyan/Megatron-Bridge/src:/root/Megatron-LM:${PYTHONPATH:-}"

HF_MODEL_PATH="${HF_MODEL_PATH:-/home/jovyan/models/gemma-3-1b-pt-hf}"
SAVE_PATH="${SAVE_PATH:-/home/jovyan/models/gemma-3-1b-trained}"
FINAL_HF_PATH="${FINAL_HF_PATH:-/home/jovyan/models/gemma-3-1b-final-hf}"
BRIDGE_DIR="${BRIDGE_DIR:-/home/jovyan/Megatron-Bridge}"

python "${BRIDGE_DIR}/examples/conversion/convert_checkpoints.py" export \
    --hf-model "${HF_MODEL_PATH}" \
    --megatron-path "${SAVE_PATH}" \
    --hf-path "${FINAL_HF_PATH}" \
    --trust-remote-code
