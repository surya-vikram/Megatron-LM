#!/bin/bash
set -euo pipefail

export PYTHONPATH="/home/jovyan/Megatron-Bridge/src:/root/Megatron-LM:${PYTHONPATH:-}"

HF_MODEL_PATH="${HF_MODEL_PATH:-/home/jovyan/models/gemma-3-1b-pt-hf}"
MCORE_CHECKPOINT="${MCORE_CHECKPOINT:-/home/jovyan/models/gemma-3-1b-pt-mcore}"
BRIDGE_DIR="${BRIDGE_DIR:-/home/jovyan/Megatron-Bridge}"

if [ ! -d "${HF_MODEL_PATH}" ]; then
    hf download google/gemma-3-1b-pt --local-dir "${HF_MODEL_PATH}"
fi

if [ "${FORCE_REIMPORT:-0}" = "1" ]; then
    rm -rf "${MCORE_CHECKPOINT}"
fi

RUN_CONFIG_PATH="${MCORE_CHECKPOINT}/run_config.yaml"
if [ ! -f "${RUN_CONFIG_PATH}" ]; then
    if [ -d "${MCORE_CHECKPOINT}" ]; then
        RUN_CONFIG_PATH=$(find "${MCORE_CHECKPOINT}" -path '*/run_config.yaml' | head -n 1 || true)
    else
        RUN_CONFIG_PATH=""
    fi
fi

if [ -z "${RUN_CONFIG_PATH}" ] || [ ! -f "${RUN_CONFIG_PATH}" ]; then
    python "${BRIDGE_DIR}/examples/conversion/convert_checkpoints.py" import \
        --hf-model "${HF_MODEL_PATH}" \
        --megatron-path "${MCORE_CHECKPOINT}" \
        --torch-dtype bfloat16 \
        --trust-remote-code
fi
