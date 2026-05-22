#!/bin/bash
set -euo pipefail

# train.sh: Modular CPT/SFT Training Interface for Gemma 3
# Optimized for Single-GPU 1B, 4B, 12B runs.

# Performance & Stability Tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

usage() {
    echo "Usage: ./train.sh --mode <cpt|sft> --model-size <1b|4b|12b> --hf-model <path> --mcore-path <path> --data-path <path> --save-path <path> [options]"
    echo "Options:"
    echo "  --iters <int>          Number of iterations (default: 20)"
    echo "  --seq-len <int>        Sequence length (default: 2048)"
    echo "  --lr <float>           Learning rate (default: 5e-6)"
    echo "  --master-port <int>    Distributed port (default: 6000)"
    exit 1
}

MODE=""
MODEL_SIZE=""
HF_MODEL=""
MCORE_PATH=""
DATA_PATH=""
SAVE_PATH=""
ITERS=20
SEQ_LEN=2048
LR="5e-6"
MASTER_PORT=6000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --hf-model) HF_MODEL="$2"; shift 2 ;;
        --mcore-path) MCORE_PATH="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$MODE" || -z "$MODEL_SIZE" || -z "$HF_MODEL" || -z "$MCORE_PATH" || -z "$DATA_PATH" || -z "$SAVE_PATH" ]] && usage

# Specific Config Values for Gemma 3 Models
case "$MODEL_SIZE" in
    1b)
        NUM_LAYERS=26
        HIDDEN_SIZE=1152
        NUM_ATTN_HEADS=8
        NUM_QUERY_GROUPS=1
        FFN_HIDDEN_SIZE=6912
        WINDOW_SIZE=512
        ;;
    4b)
        NUM_LAYERS=34
        HIDDEN_SIZE=2560
        NUM_ATTN_HEADS=8
        NUM_QUERY_GROUPS=4
        FFN_HIDDEN_SIZE=10240
        WINDOW_SIZE=1024
        ;;
    12b)
        NUM_LAYERS=48
        HIDDEN_SIZE=3840
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=8
        FFN_HIDDEN_SIZE=15360
        WINDOW_SIZE=1024
        ;;
    *)
        echo "Invalid model size: $MODEL_SIZE. Choose 1b, 4b, or 12b."
        exit 1
        ;;
esac

# Get Vocab Size from config
VOCAB_SIZE=$(python3 -c "import json, sys; print(json.load(open('$HF_MODEL/config.json')).get('vocab_size', 262144))")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FINAL_SAVE_PATH="${SAVE_PATH}/${MODE}_${TIMESTAMP}"
LOG_DIR="/home/jovyan/logs/gemma3/${MODE}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "--- Launching $MODE Training for Gemma 3 ($MODEL_SIZE) --- "
echo "Architecture: $NUM_LAYERS Layers, $HIDDEN_SIZE Hidden, $NUM_ATTN_HEADS Heads, $NUM_QUERY_GROUPS KV-Groups"
echo "Logs: $LOG_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTN_HEADS"
    --num-query-groups "$NUM_QUERY_GROUPS"
    --group-query-attention
    --kv-channels 256
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --seq-length "$SEQ_LEN"
    --max-position-embeddings 32768
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --disable-bias-linear
    --apply-layernorm-1p
    --no-rope-fusion
    --transformer-impl transformer_engine
    --attention-backend flash
    --attention-softmax-in-fp32
    --window-size "$WINDOW_SIZE,$WINDOW_SIZE"
    --vocab-size "$VOCAB_SIZE"
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --train-iters "$ITERS"
    --lr "$LR"
    --lr-decay-style cosine
    --optimizer adam
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --main-params-dtype fp16
    --exp-avg-dtype fp16
    --exp-avg-sq-dtype fp16
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --empty-unused-memory-level 1
    --manual-gc
    --manual-gc-interval 5
    --recompute-activations
    --recompute-granularity full
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-load-optim
    --no-load-rng
)

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$HF_MODEL"
    --split 100,0,0
)

[[ "$MODE" == "sft" ]] && DATA_ARGS+=(--sft)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval "$ITERS"
    --eval-interval 1000
    --eval-iters 0
    --tensorboard-dir "${FINAL_SAVE_PATH}/tensorboard"
    --log-throughput
    --log-params-norm
)

python3 -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    examples/gemma3/utils/pretrain_gemma3_mcore.py \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    --load "$MCORE_PATH" \
    --save "$FINAL_SAVE_PATH" 2>&1 | tee "$LOG_DIR/training.log"
