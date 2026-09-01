#!/bin/bash
set -euo pipefail

# User inputs.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$SCRIPT_DIR/data/pretrain/overfit_text_document}"
VALID_DATA_PATH="${VALID_DATA_PATH:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/datasets/megadata/hf_models/chimera-10b}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_440b_runs}"
INTRA_DOC_MASKING="${INTRA_DOC_MASKING:-false}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
TRAIN_ITERS="${TRAIN_ITERS:-26855}"
OPTIMIZER="${OPTIMIZER:-adam}"
MUON_NUM_NS_STEPS="${MUON_NUM_NS_STEPS:-6}"

# Distributed launch settings.
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29591}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
source "$SCRIPT_DIR/context_phase.sh"
if [[ "$CONTEXT_PHASE" != 8k ]]; then
    echo "train_440B.sh is the base 8K launcher; use context_extend.sh for later phases" >&2
    exit 1
fi

# WSD ratios: warm up for 0.3%, hold through 75%, then linearly anneal to zero.
LR_WARMUP_ITERS=$((TRAIN_ITERS * 3 / 1000))
LR_WSD_ANNEAL_START_ITERS=$((TRAIN_ITERS * 3 / 4))
LR_WSD_DECAY_ITERS=$((TRAIN_ITERS - LR_WSD_ANNEAL_START_ITERS))
TOKENS_PER_ITER=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TOTAL_TOKENS=$((TOKENS_PER_ITER * TRAIN_ITERS))

RUN_STAMP="${RUN_STAMP:-$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUNS_ROOT}/${RUN_STAMP}"
SAVE_PATH="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATA_CACHE_PATH="${RUN_DIR}/data_cache"
LOG_DIR="${RUN_DIR}/logs"

[[ -f "${TRAIN_DATA_PATH}.bin" ]] || { echo "Missing train data bin: ${TRAIN_DATA_PATH}.bin"; exit 1; }
[[ -f "${TRAIN_DATA_PATH}.idx" ]] || { echo "Missing train data idx: ${TRAIN_DATA_PATH}.idx"; exit 1; }
if [[ -n "$VALID_DATA_PATH" ]]; then
    [[ -f "${VALID_DATA_PATH}.bin" ]] || { echo "Missing validation data bin: ${VALID_DATA_PATH}.bin"; exit 1; }
    [[ -f "${VALID_DATA_PATH}.idx" ]] || { echo "Missing validation data idx: ${VALID_DATA_PATH}.idx"; exit 1; }
fi
[[ -d "$TOKENIZER_MODEL" || -f "$TOKENIZER_MODEL" ]] || { echo "Missing tokenizer model: $TOKENIZER_MODEL"; exit 1; }
if [[ -n "$LOAD_CHECKPOINT" && ! -f "$LOAD_CHECKPOINT/latest_checkpointed_iteration.txt" ]]; then
    echo "Invalid checkpoint root: $LOAD_CHECKPOINT"
    exit 1
fi
if [[ "$GLOBAL_BATCH_SIZE" -ne 2000 || "$TRAIN_ITERS" -ne 26855 ]]; then
    echo "Warning: the nominal 440B token budget is calibrated for GLOBAL_BATCH_SIZE=2000 and TRAIN_ITERS=26855; current settings produce $TOTAL_TOKENS tokens." >&2
fi

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
    --num-experts 32
    --moe-layer-freq "[0]*2+[1]*23"
    --moe-router-topk 4
    --moe-ffn-hidden-size 2048
    --moe-router-load-balancing-type quantile_balancing
    --moe-aux-loss-coeff 0.0
    --moe-qb-num-bins 1000
    --moe-qb-ema-decay 0.0
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.0
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-z-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-fusion
    --moe-per-layer-logging
    --moe-router-balance-logging-interval 1
)

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

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr 1e-3
    --min-lr 1e-6
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --attention-softmax-in-fp32
    --manual-gc
    --manual-gc-interval 100
    --use-distributed-optimizer
    --optimizer "$OPTIMIZER"
    --cuda-graph-impl transformer_engine
    --cuda-graph-modules attn
    --overlap-grad-reduce
    # --overlap-param-gather
)

if [[ "${FUSED_LINEAR_CROSS_ENTROPY:-true}" == "true" ]]; then
    TRAINING_ARGS+=(--fused-linear-cross-entropy)
fi

if [[ "$OPTIMIZER" == "muon" || "$OPTIMIZER" == "adaptive_muon" ]]; then
    TRAINING_ARGS+=(--muon-num-ns-steps "$MUON_NUM_NS_STEPS")
    OPTIMIZER_SUMMARY="$OPTIMIZER ns_steps=$MUON_NUM_NS_STEPS"
else
    TRAINING_ARGS+=(
        --use-precision-aware-optimizer
        --main-params-dtype fp32
        --main-grads-dtype fp32
        --exp-avg-dtype fp32
        --exp-avg-sq-dtype fp32
    )
    OPTIMIZER_SUMMARY="$OPTIMIZER beta1=0.9 beta2=0.95 eps=1e-8 wd=0.1 states=fp32"
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --expert-model-parallel-size "$EP_SIZE"
    --context-parallel-size "$CP_SIZE"
)

LOGGING_ARGS=(
    --save "$SAVE_PATH"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --save-interval 5000
    --eval-interval 1000
    --log-interval 1
    --log-throughput
    --exit-signal-handler
    --ckpt-format torch_dist
)
if [[ -n "$VALID_DATA_PATH" ]]; then
    LOGGING_ARGS+=(--eval-iters 4)
else
    LOGGING_ARGS+=(--eval-iters 0)
fi
if [[ -n "$LOAD_CHECKPOINT" ]]; then
    LOGGING_ARGS+=(
        --load "$LOAD_CHECKPOINT"
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
TENSORBOARD_DIR=${TENSORBOARD_DIR}
DATA_CACHE_PATH=${DATA_CACHE_PATH}
LOG_DIR=${LOG_DIR}
GPUS_PER_NODE=${GPUS_PER_NODE}
NNODES=${NNODES}
NODE_RANK=${NODE_RANK}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
INTRA_DOC_MASKING=${INTRA_DOC_MASKING}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT}
TRAIN_ITERS=${TRAIN_ITERS}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS}
LR_WSD_ANNEAL_START_ITERS=${LR_WSD_ANNEAL_START_ITERS}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS}
OPTIMIZER=${OPTIMIZER}
MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS}
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

    cp "$0" "${RUN_DIR}/train_440B.sh"
fi

if [[ -n "$LOAD_CHECKPOINT" ]]; then
    echo "Resuming Chimera 440B-token pretraining from $LOAD_CHECKPOINT"
else
    echo "Running Chimera 440B-token random-init pretraining"
fi
echo "  Run dir:          $RUN_DIR"
echo "  Train prefix:     $TRAIN_DATA_PATH"
echo "  Valid prefix:     ${VALID_DATA_PATH:-disabled}"
echo "  Tokenizer:        $TOKENIZER_MODEL"
echo "  GPUs per node:    $GPUS_PER_NODE"
echo "  Parallelism:      TP=$TP_SIZE PP=$PP_SIZE EP=$EP_SIZE ETP=1 CP=$CP_SIZE"
echo "  Architecture:     layers=25 moe_layer_freq=[0]*2+[1]*23 norm=RMSNorm qk_norm=RMSNorm swiglu=true experts=32 active=4 expert_intermediate=2048 shared_experts=0"
echo "  Router:           quantile_balancing bins=1000 ema=0 aux=0 bias_rate=0 scale=2.5 z_loss=1e-3"
echo "  Router logging:   inline interval=1 raw_expert_files=false"
echo "  Context/YaRN:     phase=$CONTEXT_PHASE max=$MAX_POSITION_EMBEDDINGS factor=$ROTARY_SCALING_FACTOR original=$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"
echo "  Seq/batch/iters:  seq=$SEQ_LENGTH micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE iters=$TRAIN_ITERS"
echo "  LR schedule:      WSD peak=1e-3 min=1e-6 warmup=$LR_WARMUP_ITERS constant_through=$LR_WSD_ANNEAL_START_ITERS decay=$LR_WSD_DECAY_ITERS style=linear"
echo "  Optimizer:        $OPTIMIZER_SUMMARY"
echo "  Token budget:     tokens_per_iter=$TOKENS_PER_ITER total_tokens=$TOTAL_TOKENS (439.99232B with defaults)"
echo "  Attention:        backend=flash external_flash_attn=false cuda_graph=TE:attn"
echo "  Intra-doc mask:   $INTRA_DOC_MASKING"
echo "  Document loss:    predict_eos=true post_eos_target=false"

exec python3 -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    examples/chimera/pretrain_chimera.py \
    --chimera-expert-tp-size 1 \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" > >(tee -a "${LOG_DIR}/train_node_${NODE_RANK}.log") 2>&1
