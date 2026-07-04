#!/bin/bash
set -euo pipefail

# Required user inputs.
DATA_PATH="${DATA_PATH:-/datasets/megadata/chimera_sft/train.jsonl}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/datasets/megadata/hf_models/chimera-10b}"
MCORE_PATH="${MCORE_PATH:-/datasets/megadata/chimera_runs/pretrain/checkpoints}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_sft_runs}"

# Distributed launch settings.
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29592}"

# Training settings.
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
LR="${LR:-5e-6}"
MIN_LR="${MIN_LR:-5e-7}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-0}"
SAVE_WEIGHTS_ONLY="${SAVE_WEIGHTS_ONLY:-false}"

RUN_STAMP="$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_ROOT}/${RUN_STAMP}"
SAVE_PATH="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATA_CACHE_PATH="${RUN_DIR}/data_cache"
LOG_DIR="${RUN_DIR}/logs"

[[ -f "$DATA_PATH" ]] || { echo "Missing SFT JSONL: $DATA_PATH"; exit 1; }
[[ -d "$TOKENIZER_MODEL" || -f "$TOKENIZER_MODEL" ]] || { echo "Missing tokenizer model: $TOKENIZER_MODEL"; exit 1; }
[[ -d "$MCORE_PATH" ]] || { echo "Missing MCore checkpoint path: $MCORE_PATH"; exit 1; }

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR" "$DATA_CACHE_PATH" "$LOG_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 25
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings 32768
    --position-embedding-type yarn
    --rotary-base 10000000
    --rotary-percent 1.0
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --make-vocab-size-divisible-by 128
    --vocab-size 50176
    --bf16
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-masked-softmax-fusion
    --tokenizer-type SFTTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --sft-tokenizer-prompt-format chimera
    --fused-residual-rmsnorm
)

MOE_ARGS=(
    --num-experts 64
    --moe-layer-freq "[0]*2+[1]*23"
    --moe-router-topk 4
    --moe-ffn-hidden-size 1024
    --moe-shared-expert-intermediate-size 1024
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.0001
    --moe-router-topk-scaling-factor 1.0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-fusion
    --moe-shared-expert-overlap
)

DATA_ARGS=(
    --data-path 1.0 "$DATA_PATH"
    --split 100,0,0
    --data-cache-path "$DATA_CACHE_PATH"
    --num-workers 8
    --sft
    --eod-mask-loss
    --no-create-attention-mask-in-dataloader
)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style cosine
    --lr-decay-iters "$LR_DECAY_ITERS"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay 0.0
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --attention-softmax-in-fp32
    --manual-gc
    --manual-gc-interval 100
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --main-params-dtype fp32
    --main-grads-dtype bf16
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
    --fused-linear-cross-entropy
    --overlap-grad-reduce
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --context-parallel-size 1
)

LOGGING_ARGS=(
    --load "$MCORE_PATH"
    --finetune
    --no-load-optim
    --no-load-rng
    --save "$SAVE_PATH"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --log-interval 1
    --log-throughput
)
if [[ "$SAVE_WEIGHTS_ONLY" == true ]]; then
    LOGGING_ARGS+=(--no-save-optim --no-save-rng)
fi

cat > "${RUN_DIR}/run_paths.env" <<EOF
DATA_PATH=${DATA_PATH}
TOKENIZER_MODEL=${TOKENIZER_MODEL}
MCORE_PATH=${MCORE_PATH}
RUN_DIR=${RUN_DIR}
SAVE_PATH=${SAVE_PATH}
EOF

cp "$0" "${RUN_DIR}/sft.sh"

echo "Running Chimera SFT"
echo "  Run dir:          $RUN_DIR"
echo "  Data JSONL:       $DATA_PATH"
echo "  Load checkpoint:  $MCORE_PATH"
echo "  Tokenizer:        $TOKENIZER_MODEL"
echo "  Seq/batch/iters:  seq=$SEQ_LENGTH micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE iters=$TRAIN_ITERS"
echo "  Packing:          disabled"

python3 -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    examples/chimera/pretrain_chimera.py \
    --chimera-expert-tp-size 1 \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/sft.log"
