#!/bin/bash
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_GRAPH_REGISTER=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# User inputs.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$SCRIPT_DIR/data/pretrain/fineweb_edu_text_document}"
VALID_DATA_PATH="${VALID_DATA_PATH:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/workspace/repos/transformers/src/transformers/models/chimera/tokenizer}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/scratch/tiny_chimera_runs}"
INTRA_DOC_MASKING="${INTRA_DOC_MASKING:-false}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
source "$SCRIPT_DIR/context_phase.sh"

# GPU launch settings.
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29591}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"

RUN_STAMP="${RUN_STAMP:-$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUNS_ROOT}/${RUN_STAMP}"
SAVE_PATH="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATA_CACHE_PATH="${RUN_DIR}/data_cache"
LOG_DIR="${RUN_DIR}/logs"

[[ -f "${TRAIN_DATA_PATH}.bin" ]] || { echo "Missing train data bin: ${TRAIN_DATA_PATH}.bin"; exit 1; }
[[ -f "${TRAIN_DATA_PATH}.idx" ]] || { echo "Missing train data idx: ${TRAIN_DATA_PATH}.idx"; exit 1; }

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR" "$DATA_CACHE_PATH" "$LOG_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# Downscaled Tiny Chimera Architecture (65M total params, ultra-lightweight for local 8GB GPU)
MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 8
    --hidden-size 512
    --ffn-hidden-size 2048
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 2
    --kv-channels 64
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type "$POSITION_EMBEDDING_TYPE"
    --rotary-base 10000000
    --rotary-percent 1.0
    --rotary-scaling-factor "$ROTARY_SCALING_FACTOR"
    --yarn-original-max-position-embeddings "$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"
    --mscale 1.0
    --mscale-all-dim 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --qk-layernorm
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
    --num-experts 8
    --moe-layer-freq "[0]*2+[1]*6"
    --moe-router-topk 2
    --moe-ffn-hidden-size 256
    --moe-router-load-balancing-type "${MOE_ROUTER_LOAD_BALANCING_TYPE:-quantile_balancing}"
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.0}"
    --moe-qb-num-bins "${MOE_QB_NUM_BINS:-1000}"
    --moe-qb-ema-decay "${MOE_QB_EMA_DECAY:-0.0}"
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate "${MOE_ROUTER_BIAS_UPDATE_RATE:-0.0}"
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-z-loss-coeff "${MOE_Z_LOSS_COEFF:-0.001}"
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-balance-logging-interval "${MOE_ROUTER_BALANCE_LOGGING_INTERVAL:-1}"
)

if [[ "${ENABLE_PER_LAYER_LOGGING:-true}" == "true" ]]; then
    MOE_ARGS+=(--moe-per-layer-logging)
fi

DATA_ARGS=(
    --train-data-path "$TRAIN_DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --num-workers 32
    --eod-mask-loss
)
if [[ -n "$VALID_DATA_PATH" ]]; then
    DATA_ARGS+=(--valid-data-path "$VALID_DATA_PATH")
fi
if [[ "$INTRA_DOC_MASKING" == true ]]; then
    DATA_ARGS+=(
        --reset-attention-mask
        --reset-position-ids
    )
else
    DATA_ARGS+=(--no-create-attention-mask-in-dataloader)
fi

OPTIMIZER="${OPTIMIZER:-adam}"

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "${TRAIN_ITERS:-100}"
    --lr 1e-3
    --min-lr 1e-4
    --lr-decay-style WSD
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-iters "${LR_WSD_DECAY_ITERS:-20}"
    --lr-warmup-iters "${LR_WARMUP_ITERS:-5}"
    --weight-decay 0.1
    --clip-grad 1.0
    --optimizer "$OPTIMIZER"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --attention-softmax-in-fp32
    --manual-gc
    --manual-gc-interval 1000
    --cuda-graph-impl transformer_engine
    --cuda-graph-modules attn
    --cuda-graph-warmup-steps 1
    --use-distributed-optimizer
)

if [[ "${FUSED_LINEAR_CROSS_ENTROPY:-true}" == "true" ]]; then
    TRAINING_ARGS+=(--fused-linear-cross-entropy)
fi

if [[ "$OPTIMIZER" == "muon" ]]; then
    TRAINING_ARGS+=(--muon-num-ns-steps "${MUON_NUM_NS_STEPS:-6}")
    if [[ -n "${MUON_SCALAR_OPTIMIZER:-}" ]]; then
        TRAINING_ARGS+=(--muon-scalar-optimizer "$MUON_SCALAR_OPTIMIZER")
    fi
fi

# adaptive_muon defaults to adamuon internally

if [[ "$OPTIMIZER" == "adam" ]]; then
    TRAINING_ARGS+=(
        --use-precision-aware-optimizer
        --main-params-dtype fp32
        --main-grads-dtype fp32
        --exp-avg-dtype fp32
        --exp-avg-sq-dtype fp32
    )
fi

if [[ "${OVERLAP_GRAD_REDUCE:-false}" == "true" ]]; then
    TRAINING_ARGS+=(--overlap-grad-reduce)
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --expert-model-parallel-size "$EP_SIZE"
    --context-parallel-size "$CP_SIZE"
)

LOGGING_ARGS=(
    --save "${SAVE_DIR:-$SAVE_PATH}"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --save-interval "${SAVE_INTERVAL:-${TRAIN_ITERS:-100}}"
    --eval-interval 10000
    --eval-iters 0
    --log-interval 1
    --log-throughput
    --exit-signal-handler
    --ckpt-format torch_dist
)
if [[ -n "${LOAD_CHECKPOINT:-}" || -n "${LOAD_DIR:-}" ]]; then
    LOGGING_ARGS+=(
        --load "${LOAD_CHECKPOINT:-${LOAD_DIR:-}}"
        --exit-on-missing-checkpoint
    )
fi

if [[ "$NODE_RANK" == 0 ]]; then
cat > "${RUN_DIR}/run_paths.env" <<EOF
TRAIN_DATA_PATH=${TRAIN_DATA_PATH}
VALID_DATA_PATH=${VALID_DATA_PATH}
TOKENIZER_MODEL=${TOKENIZER_MODEL}
RUNS_ROOT=${RUNS_ROOT}
RUN_DIR=${RUN_DIR}
SAVE_PATH=${SAVE_PATH}
TENSORBOARD_DIR=${DATA_CACHE_PATH}
LOG_DIR=${LOG_DIR}
GPUS_PER_NODE=${GPUS_PER_NODE}
NNODES=${NNODES}
NODE_RANK=${NODE_RANK}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
INTRA_DOC_MASKING=${INTRA_DOC_MASKING}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT}
TP_SIZE=${TP_SIZE}
PP_SIZE=${PP_SIZE}
EP_SIZE=${EP_SIZE}
CP_SIZE=${CP_SIZE}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
SEQ_LENGTH=${SEQ_LENGTH}
CONTEXT_PHASE=${CONTEXT_PHASE}
POSITION_EMBEDDING_TYPE=${POSITION_EMBEDDING_TYPE}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS}
ROTARY_SCALING_FACTOR=${ROTARY_SCALING_FACTOR}
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}
EOF

    cp "$0" "${RUN_DIR}/tiny_chimera.sh"
fi

if [[ -n "$LOAD_CHECKPOINT" ]]; then
    echo "Resuming Tiny Chimera pretraining from $LOAD_CHECKPOINT"
else
    echo "Running Tiny Chimera random-init pretraining"
fi
echo "  Run dir:          $RUN_DIR"
echo "  Train prefix:     $TRAIN_DATA_PATH"
echo "  Tokenizer:        $TOKENIZER_MODEL"
echo "  GPUs per node:    $GPUS_PER_NODE"
echo "  Parallelism:      TP=$TP_SIZE PP=$PP_SIZE EP=$EP_SIZE ETP=1 CP=$CP_SIZE"
echo "  Architecture:     layers=8 moe_layer_freq=[0]*2+[1]*6 hidden=512 ffn=2048 experts=8 topk=2 expert_ffn=256 shared=0 qk_norm=true"
echo "  Router:           ${MOE_ROUTER_LOAD_BALANCING_TYPE:-quantile_balancing} bins=${MOE_QB_NUM_BINS:-1000} ema=${MOE_QB_EMA_DECAY:-0.0} aux=${MOE_AUX_LOSS_COEFF:-0.0} bias_rate=${MOE_ROUTER_BIAS_UPDATE_RATE:-0.0} scale=2.5 z_loss=${MOE_Z_LOSS_COEFF:-0.001}"
echo "  Context/YaRN:     phase=$CONTEXT_PHASE max=$MAX_POSITION_EMBEDDINGS factor=$ROTARY_SCALING_FACTOR original=$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"
echo "  Seq/batch/iters:  seq=$SEQ_LENGTH micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE iters=${TRAIN_ITERS:-100}"

exec python3 -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    examples/chimera/pretrain_chimera.py \
    --chimera-expert-tp-size 1 \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@" > >(tee -a "${LOG_DIR}/train_node_${NODE_RANK}.log") 2>&1
