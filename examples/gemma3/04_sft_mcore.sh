#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./examples/gemma3/04_sft_mcore.sh --checkpoint-path <path> --data-path <jsonl> [options]

Legacy positional mode is also supported:
  ./examples/gemma3/04_sft_mcore.sh <checkpoint_path> <data_path>

Important:
  Packed SFT requires --micro-batch-size 1 in this Megatron-LM path.

Common options:
  --save-path <path>                Output checkpoint directory. Default: <checkpoint>_sft_<mode>
  --mode <name>                     Metadata label. Default: full
  --seq-length <int>                Active context length. Default: 16384
  --max-position-embeddings <int>   Model position limit. Default: 32768
  --micro-batch-size <int>          Must stay 1 for packed SFT.
  --global-batch-size <int>         Default: 8
  --train-iters <int>               Default: 20
  --lr <float>                      Default: 5e-6
  --min-lr <float>                  Default: same as --lr
  --weight-decay <float>            Default: 0.0
  --clip-grad <float>               Default: 1.0
  --num-workers <int>               Default: 8
  --save-interval <int>             Default: 50
  --eval-interval <int>             Default: 1000
  --eval-iters <int>                Default: 0
  --log-interval <int>              Default: 1
  --master-port <int>               Default: 6000
  --resume                          Resume SFT optimizer/rng state from save-path.
  --tokenizer-model <model>         Default: google/gemma-3-1b-it
EOF
}

CHECKPOINT_PATH=""
DATA_PATH=""
SAVE_PATH=""
MODE="full"
SEQ_LENGTH=16384
MAX_POSITION_EMBEDDINGS=32768
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
TRAIN_ITERS=20
LR="5e-6"
MIN_LR=""
WEIGHT_DECAY="0.0"
CLIP_GRAD="1.0"
NUM_WORKERS=8
SAVE_INTERVAL=50
EVAL_INTERVAL=1000
EVAL_ITERS=0
LOG_INTERVAL=1
MASTER_PORT=6000
TOKENIZER_MODEL="google/gemma-3-1b-it"
RESUME=0

if [[ $# -ge 2 && "${1:-}" != --* ]]; then
    CHECKPOINT_PATH="$1"
    DATA_PATH="$2"
    shift 2
fi

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
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --max-position-embeddings)
            MAX_POSITION_EMBEDDINGS="$2"
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
        --min-lr)
            MIN_LR="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --clip-grad)
            CLIP_GRAD="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --eval-iters)
            EVAL_ITERS="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --tokenizer-model)
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
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

if [[ -z "$CHECKPOINT_PATH" || -z "$DATA_PATH" ]]; then
    echo "Error: checkpoint path and data path are required."
    usage
    exit 1
fi

if [[ "$MICRO_BATCH_SIZE" -ne 1 ]]; then
    echo "Error: packed SFT requires --micro-batch-size 1 in this Megatron-LM path."
    exit 1
fi

if [[ -z "$MIN_LR" ]]; then
    MIN_LR="$LR"
fi

if [[ -z "$SAVE_PATH" ]]; then
    SAVE_PATH="${CHECKPOINT_PATH}_sft_${MODE}"
fi

mkdir -p "$SAVE_PATH"
mkdir -p "${SAVE_PATH}/tensorboard"

python3 - <<PY
import json
from pathlib import Path

payload = {
    "mode": "${MODE}",
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "data_path": "${DATA_PATH}",
    "save_path": "${SAVE_PATH}",
    "seq_length": int("${SEQ_LENGTH}"),
    "max_position_embeddings": int("${MAX_POSITION_EMBEDDINGS}"),
    "micro_batch_size": int("${MICRO_BATCH_SIZE}"),
    "global_batch_size": int("${GLOBAL_BATCH_SIZE}"),
    "train_iters": int("${TRAIN_ITERS}"),
    "lr": "${LR}",
    "min_lr": "${MIN_LR}",
    "weight_decay": "${WEIGHT_DECAY}",
    "clip_grad": "${CLIP_GRAD}",
    "num_workers": int("${NUM_WORKERS}"),
    "save_interval": int("${SAVE_INTERVAL}"),
    "eval_interval": int("${EVAL_INTERVAL}"),
    "eval_iters": int("${EVAL_ITERS}"),
    "resume": bool(int("${RESUME}")),
    "tokenizer_model": "${TOKENIZER_MODEL}",
}
Path("${SAVE_PATH}/run_config.json").write_text(json.dumps(payload, indent=2))
PY

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --master_addr localhost
    --master_port "$MASTER_PORT"
    --node_rank 0
)

NUM_LAYERS=26
HIDDEN_SIZE=1152
NUM_ATTN_HEADS=4
NUM_QUERY_GROUPS=1
FFN_HIDDEN_SIZE=6912
WINDOW_SIZE=512
VOCAB_SIZE=262144

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTN_HEADS"
    --num-query-groups "$NUM_QUERY_GROUPS"
    --group-query-attention
    --kv-channels 256
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
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
    --lr-decay-style constant
    --min-lr "$MIN_LR"
    --weight-decay "$WEIGHT_DECAY"
    --clip-grad "$CLIP_GRAD"
    --bf16
    --num-workers "$NUM_WORKERS"
    --recompute-activations
)

DATA_ARGS=(
    --sft
    --data-path "$DATA_PATH"
    --tokenizer-type SFTTokenizer
    --sft-tokenizer-prompt-format gemma3
    --tokenizer-model "$TOKENIZER_MODEL"
    --no-create-attention-mask-in-dataloader
    --vocab-size "$VOCAB_SIZE"
    --split 100,0,0
)

LOGGING_ARGS=(
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_PATH"
    --tensorboard-dir "${SAVE_PATH}/tensorboard"
    --log-interval "$LOG_INTERVAL"
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --log-throughput
)

if [[ "$RESUME" -eq 0 ]]; then
    LOGGING_ARGS+=(--no-load-optim --no-load-rng)
fi

python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" examples/gemma3/pretrain_gemma3_mcore.py \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"
