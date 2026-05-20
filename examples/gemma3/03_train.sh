#!/bin/bash
set -euo pipefail

export PYTHONPATH="/home/jovyan/Megatron-Bridge/src:/root/Megatron-LM:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/jovyan/data/corpus}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DATA_DIR}/corpus_data}"
DATA_PATH="${DATA_PATH:-${OUTPUT_PREFIX}_text_document}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/home/jovyan/models/gemma-3-1b-pt-hf}"
MCORE_CHECKPOINT="${MCORE_CHECKPOINT:-/home/jovyan/models/gemma-3-1b-pt-mcore}"
SAVE_PATH="${SAVE_PATH:-/home/jovyan/models/gemma-3-1b-trained}"

SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-6}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${TRAIN_ITERS}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-0}"

mkdir -p "${SAVE_PATH}"

if [ "${LR_WARMUP_ITERS}" -ge "${LR_DECAY_ITERS}" ]; then
    if [ "${LR_DECAY_ITERS}" -gt 1 ]; then
        LR_WARMUP_ITERS=$((LR_DECAY_ITERS - 1))
    else
        LR_WARMUP_ITERS=0
    fi
fi

export DATA_PATH HF_MODEL_PATH MCORE_CHECKPOINT SAVE_PATH
export SEQ_LENGTH MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TRAIN_ITERS
export LEARNING_RATE MIN_LEARNING_RATE LR_WARMUP_ITERS LR_DECAY_ITERS
export SAVE_INTERVAL EVAL_INTERVAL EVAL_ITERS

torchrun --nproc_per_node=1 /root/Megatron-LM/examples/gemma3/train_bridge.py
