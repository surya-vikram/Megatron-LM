#!/bin/bash
set -euo pipefail

# User inputs.
DATA_PATH="${DATA_PATH:-/datasets/megadata/chimera/overfit_doc_text_document}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/datasets/megadata/hf_models/chimera-10b}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_runs}"
INTRA_DOC_MASKING="${INTRA_DOC_MASKING:-false}"

# Distributed launch settings.
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29591}"

RUN_STAMP="$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_ROOT}/${RUN_STAMP}"
SAVE_PATH="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATA_CACHE_PATH="${RUN_DIR}/data_cache"
LOG_DIR="${RUN_DIR}/logs"

[[ -f "${DATA_PATH}.bin" ]] || { echo "Missing data bin: ${DATA_PATH}.bin"; exit 1; }
[[ -f "${DATA_PATH}.idx" ]] || { echo "Missing data idx: ${DATA_PATH}.idx"; exit 1; }
[[ -d "$TOKENIZER_MODEL" || -f "$TOKENIZER_MODEL" ]] || { echo "Missing tokenizer model: $TOKENIZER_MODEL"; exit 1; }

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
    --seq-length 8192
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
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
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
    # --moe-per-layer-logging
)

DATA_ARGS=(
    --data-path 1.0 "$DATA_PATH"
    --split 100,0,0
    --data-cache-path "$DATA_CACHE_PATH"
    --num-workers 8
)
if [[ "$INTRA_DOC_MASKING" == true ]]; then
    DATA_ARGS+=(
        --reset-attention-mask
        --reset-position-ids
    )
else
    DATA_ARGS+=(--no-create-attention-mask-in-dataloader)
fi

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 16
    --train-iters 10000
    --lr 2e-4
    --min-lr 2e-5
    --lr-decay-style cosine
    --lr-decay-iters 10000
    --lr-warmup-iters 0
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
    --cuda-graph-impl transformer_engine
    --cuda-graph-modules attn
    --overlap-grad-reduce
    # --overlap-param-gather
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --context-parallel-size 1
)

LOGGING_ARGS=(
    --save "$SAVE_PATH"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --save-interval 1000
    --eval-interval 1000
    --eval-iters 0
    --log-interval 1
    --log-throughput
)

cat > "${RUN_DIR}/run_paths.env" <<EOF
DATA_PATH=${DATA_PATH}
TOKENIZER_MODEL=${TOKENIZER_MODEL}
RUNS_ROOT=${RUNS_ROOT}
RUN_DIR=${RUN_DIR}
SAVE_PATH=${SAVE_PATH}
TENSORBOARD_DIR=${TENSORBOARD_DIR}
DATA_CACHE_PATH=${DATA_CACHE_PATH}
LOG_DIR=${LOG_DIR}
GPUS_PER_NODE=${GPUS_PER_NODE}
NNODES=${NNODES}
NODE_RANK=${NODE_RANK}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
INTRA_DOC_MASKING=${INTRA_DOC_MASKING}
EOF

cp "$0" "${RUN_DIR}/train.sh"

echo "Running Chimera random-init pretraining"
echo "  Run dir:          $RUN_DIR"
echo "  Data prefix:      $DATA_PATH"
echo "  Tokenizer:        $TOKENIZER_MODEL"
echo "  GPUs per node:    $GPUS_PER_NODE"
echo "  Parallelism:      TP=1 PP=1 EP=1 ETP=1 CP=1"
echo "  Architecture:     layers=25 moe_layer_freq=[0]*2+[1]*23"
echo "  Seq/batch/iters:  seq=8192 micro=4 global=16 iters=10000"
echo "  Attention:        backend=flash external_flash_attn=false cuda_graph=TE:attn"
echo "  Intra-doc mask:   $INTRA_DOC_MASKING"

python3 -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    examples/chimera/pretrain_chimera.py \
    --chimera-expert-tp-size 1 \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
