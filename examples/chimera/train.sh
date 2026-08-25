#!/bin/bash
set -euo pipefail

# User inputs.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$SCRIPT_DIR/data/pretrain/overfit_text_document}"
VALID_DATA_PATH="${VALID_DATA_PATH:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/datasets/megadata/hf_models/chimera-10b}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_runs}"
INTRA_DOC_MASKING="${INTRA_DOC_MASKING:-false}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"

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
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-576}"

# Production defaults can be overridden for documented smoke/overfit runs
# without editing this canonical launcher.
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
TRAIN_ITERS="${TRAIN_ITERS:-423856}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-minus_sqrt}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-84771}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-1695}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"

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
if [[ "$GLOBAL_BATCH_SIZE" -ne 576 ]]; then
    echo "Warning: the 2T iteration and WSD schedule is calibrated for GLOBAL_BATCH_SIZE=576; recalculate schedule iterations for $GLOBAL_BATCH_SIZE." >&2
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
    --max-position-embeddings 8192
    --position-embedding-type yarn
    --rotary-base 10000000
    --rotary-percent 1.0
    --rotary-scaling-factor 1.0
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
    --num-workers 8
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
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay "$WEIGHT_DECAY"
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --attention-softmax-in-fp32
    --manual-gc
    --manual-gc-interval 100
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --main-params-dtype fp32
    --main-grads-dtype "$MAIN_GRADS_DTYPE"
    --exp-avg-dtype "$EXP_AVG_DTYPE"
    --exp-avg-sq-dtype "$EXP_AVG_SQ_DTYPE"
    --fused-linear-cross-entropy
    --cuda-graph-impl transformer_engine
    --cuda-graph-modules attn
    --overlap-grad-reduce
    # --overlap-param-gather
)
if [[ "$LR_DECAY_STYLE" == "WSD" ]]; then
    TRAINING_ARGS+=(
        --lr-wsd-decay-style "$LR_WSD_DECAY_STYLE"
        --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    )
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
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
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
TP_SIZE=${TP_SIZE}
PP_SIZE=${PP_SIZE}
EP_SIZE=${EP_SIZE}
CP_SIZE=${CP_SIZE}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
SEQ_LENGTH=${SEQ_LENGTH}
TRAIN_ITERS=${TRAIN_ITERS}
LR=${LR}
MIN_LR=${MIN_LR}
LR_DECAY_STYLE=${LR_DECAY_STYLE}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS}
WEIGHT_DECAY=${WEIGHT_DECAY}
MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE}
EXP_AVG_DTYPE=${EXP_AVG_DTYPE}
EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE}
EOF

    cp "$0" "${RUN_DIR}/train.sh"
fi

if [[ -n "$LOAD_CHECKPOINT" ]]; then
    echo "Resuming Chimera pretraining from $LOAD_CHECKPOINT"
else
    echo "Running Chimera random-init pretraining"
fi
echo "  Run dir:          $RUN_DIR"
echo "  Train prefix:     $TRAIN_DATA_PATH"
echo "  Valid prefix:     ${VALID_DATA_PATH:-disabled}"
echo "  Tokenizer:        $TOKENIZER_MODEL"
echo "  GPUs per node:    $GPUS_PER_NODE"
echo "  Parallelism:      TP=$TP_SIZE PP=$PP_SIZE EP=$EP_SIZE ETP=1 CP=$CP_SIZE"
echo "  Architecture:     layers=25 moe_layer_freq=[0]*2+[1]*23 hidden=2048 experts=32 topk=4 expert_ffn=2048 shared=0 qk_norm=true"
echo "  Router:           quantile_balancing bins=1000 ema=0 aux=0 bias_rate=0 scale=2.5 z_loss=1e-3"
echo "  Router logging:   inline interval=1 raw_expert_files=false"
echo "  Seq/batch/iters:  seq=$SEQ_LENGTH max_position=8192 micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE iters=$TRAIN_ITERS"
echo "  LR schedule:      $LR_DECAY_STYLE peak=$LR min=$MIN_LR warmup=$LR_WARMUP_ITERS wsd_decay=$LR_WSD_DECAY_ITERS wsd_style=$LR_WSD_DECAY_STYLE"
echo "  Optimizer:        AdamW beta1=0.9 beta2=0.95 eps=1e-8 wd=$WEIGHT_DECAY main_params=fp32 main_grads=$MAIN_GRADS_DTYPE exp_avg=$EXP_AVG_DTYPE exp_avg_sq=$EXP_AVG_SQ_DTYPE"
echo "  Production budget: approximately 2.0T at global_batch=576 with the default schedule"
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
