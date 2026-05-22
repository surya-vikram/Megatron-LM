#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./examples/gemma3/03_train.sh --checkpoint-path <path> --data-path <prefix> --hf-model-path <path> [options]

Common options:
  --save-path <path>                Output checkpoint directory.
  --seq-length <int>                Default: 2048
  --micro-batch-size <int>          Default: 1
  --global-batch-size <int>         Default: 1
  --train-iters <int>               Default: 100
  --lr <float>                      Default: 1e-5
  --save-interval <int>             Default: 100
  --master-port <int>               Default: 6000
EOF
}

CHECKPOINT_PATH=""
DATA_PATH=""
HF_MODEL_PATH=""
SAVE_PATH=""
SEQ_LENGTH=2048
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
TRAIN_ITERS=100
LR="1e-5"
SAVE_INTERVAL=100
MASTER_PORT=6000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --hf-model-path)
            HF_MODEL_PATH="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --global-batch-size)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --train-iters)
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CHECKPOINT_PATH" || -z "$DATA_PATH" || -z "$HF_MODEL_PATH" ]]; then
    echo "Error: checkpoint-path, data-path, and hf-model-path are required."
    usage
    exit 1
fi

if [[ -z "$SAVE_PATH" ]]; then
    SAVE_PATH="${CHECKPOINT_PATH}_trained"
fi

mkdir -p "$SAVE_PATH"

# Dynamic Architecture Inference
if [[ -d "$HF_MODEL_PATH" && -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "Inferring architecture from $HF_MODEL_PATH/config.json ..."
    read -r NUM_LAYERS HIDDEN_SIZE NUM_ATTN_HEADS NUM_QUERY_GROUPS FFN_HIDDEN_SIZE WINDOW_SIZE VOCAB_SIZE < <(python3 -c "
import json
from pathlib import Path
config = json.loads(Path('$HF_MODEL_PATH/config.json').read_text())
print(f\"{config.get('num_hidden_layers', 26)} {config.get('hidden_size', 1152)} {config.get('num_attention_heads', 8)} {config.get('num_key_value_heads', 1)} {config.get('intermediate_size', 6912)} {config.get('sliding_window', 512)} {config.get('vocab_size', 262144)}\")
")
else
    echo "Warning: $HF_MODEL_PATH/config.json not found. Falling back to 1B defaults."
    NUM_LAYERS=26
    HIDDEN_SIZE=1152
    NUM_ATTN_HEADS=4
    NUM_QUERY_GROUPS=1
    FFN_HIDDEN_SIZE=6912
    WINDOW_SIZE=512
    VOCAB_SIZE=262144
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --master_addr localhost
    --master_port "$MASTER_PORT"
    --node_rank 0
)

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTN_HEADS"
    --num-query-groups "$NUM_QUERY_GROUPS"
    --group-query-attention
    --kv-channels 256
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --seq-length "$SEQ_LENGTH"
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
)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr "$LR"
    --lr-decay-style cosine
    --bf16
    --recompute-activations
)

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$HF_MODEL_PATH"
    --vocab-size "$VOCAB_SIZE"
    --split 100,0,0
)

LOGGING_ARGS=(
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_PATH"
    --tensorboard-dir "${SAVE_PATH}/tensorboard"
    --log-interval 1
    --save-interval "$SAVE_INTERVAL"
    --eval-interval 1000
    --eval-iters 0
    --log-throughput
)

python3 -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" examples/gemma3/pretrain_gemma3_mcore.py \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    --no-load-optim --no-load-rng
