#!/bin/bash
set -euo pipefail

# User inputs.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$SCRIPT_DIR/data/pretrain/overfit_text_document}"
VALID_DATA_PATH="${VALID_DATA_PATH:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/datasets/megadata/hf_models/chimera-10b}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_runs}"
# Chimera pretraining always uses a standard causal mask across packed documents.
INTRA_DOC_MASKING=false
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
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-576}"

# Production defaults can be overridden for documented smoke/overfit runs
# without editing this canonical launcher.
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
source "$SCRIPT_DIR/context_phase.sh"
source "$SCRIPT_DIR/schedule_helpers.sh"
CHIMERA_CONTEXT_EXTENSION="${CHIMERA_CONTEXT_EXTENSION:-false}"
TRAIN_TOKENS="${TRAIN_TOKENS:-4000000000000}"
TOKENS_PER_ITER=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
if [[ -z "${TRAIN_ITERS:-}" ]]; then
    chimera_require_positive_integer TRAIN_TOKENS "$TRAIN_TOKENS"
    TRAIN_ITERS=$(chimera_ceil_div "$TRAIN_TOKENS" "$TOKENS_PER_ITER")
fi
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-2e-5}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-linear}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
if [[ "$CHIMERA_CONTEXT_EXTENSION" == true ]]; then
    OPTIMIZER="${OPTIMIZER:-adam}"
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-fused}"
    QK_CLIP="${QK_CLIP:-true}"
    LOG_MAX_ATTENTION_LOGIT="${LOG_MAX_ATTENTION_LOGIT:-true}"
else
    OPTIMIZER="${OPTIMIZER:-muon}"
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-fused}"
    QK_CLIP="${QK_CLIP:-true}"
    LOG_MAX_ATTENTION_LOGIT="${LOG_MAX_ATTENTION_LOGIT:-true}"
fi
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NUM_NS_STEPS="${MUON_NUM_NS_STEPS:-5}"
MUON_SCALE_MODE="${MUON_SCALE_MODE:-spectral}"
MUON_EXTRA_SCALE_FACTOR="${MUON_EXTRA_SCALE_FACTOR:-0.2}"
MUON_SCALAR_OPTIMIZER="${MUON_SCALAR_OPTIMIZER:-adam}"
MUON_SPLIT_QKV="${MUON_SPLIT_QKV:-true}"
MUON_NESTEROV="${MUON_NESTEROV:-false}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
QK_LAYERNORM="${QK_LAYERNORM:-true}"
QK_CLIP_THRESHOLD="${QK_CLIP_THRESHOLD:-100.0}"
QK_CLIP_ALPHA="${QK_CLIP_ALPHA:-0.5}"
APPLY_ROPE_FUSION="${APPLY_ROPE_FUSION:-true}"
FUSED_LINEAR_CROSS_ENTROPY="${FUSED_LINEAR_CROSS_ENTROPY:-true}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_ITERS="${EVAL_ITERS:-4}"
NUM_WORKERS="${NUM_WORKERS:-32}"
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"
MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"

[[ "$LR_DECAY_STYLE" == cosine || "$LR_DECAY_STYLE" == WSD ]] || {
    echo "Unsupported LR_DECAY_STYLE=$LR_DECAY_STYLE; expected cosine or WSD" >&2
    exit 1
}
chimera_resolve_schedule

[[ "$OPTIMIZER" == adam || "$OPTIMIZER" == muon ]] || {
    echo "Unsupported OPTIMIZER=$OPTIMIZER; expected adam or muon" >&2
    exit 1
}
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
if [[ "$CHIMERA_CONTEXT_EXTENSION" == true ]]; then
    [[ -n "$LOAD_CHECKPOINT" ]] || { echo "Context extension requires LOAD_CHECKPOINT"; exit 1; }
    [[ "$SEQ_LENGTH" -eq "$MAX_POSITION_EMBEDDINGS" ]] || {
        echo "Context extension requires SEQ_LENGTH=$MAX_POSITION_EMBEDDINGS for phase $CONTEXT_PHASE"; exit 1;
    }
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
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --make-vocab-size-divisible-by 128
    --vocab-size 50176
    --bf16
    --attention-backend "$ATTENTION_BACKEND"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-masked-softmax-fusion
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --fused-residual-rmsnorm
)
if [[ "$APPLY_ROPE_FUSION" == true ]]; then
    MODEL_ARGS+=(--apply-rope-fusion)
fi
if [[ "$QK_LAYERNORM" == true ]]; then
    MODEL_ARGS+=(--qk-layernorm)
fi

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
    --num-workers "$NUM_WORKERS"
    --eod-mask-loss
)
if [[ -n "$VALID_DATA_PATH" ]]; then
    DATA_ARGS+=(--valid-data-path "$VALID_DATA_PATH")
fi
DATA_ARGS+=(--no-create-attention-mask-in-dataloader)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay "$WEIGHT_DECAY"
    --clip-grad "$CLIP_GRAD"
    --optimizer "$OPTIMIZER"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
    --adam-eps "$ADAM_EPS"
    --attention-softmax-in-fp32
    --manual-gc
    --manual-gc-interval 100
    --use-distributed-optimizer
    --cuda-graph-impl transformer_engine
    --cuda-graph-modules attn
    --overlap-grad-reduce
    # --overlap-param-gather
)
if [[ "$QK_CLIP" == true ]]; then
    TRAINING_ARGS+=(
        --qk-clip
        --qk-clip-threshold "$QK_CLIP_THRESHOLD"
        --qk-clip-alpha "$QK_CLIP_ALPHA"
    )
fi
if [[ "$LOG_MAX_ATTENTION_LOGIT" == true ]]; then
    TRAINING_ARGS+=(--log-max-attention-logit)
fi
if [[ "$FUSED_LINEAR_CROSS_ENTROPY" == true ]]; then
    TRAINING_ARGS+=(--fused-linear-cross-entropy)
fi
if [[ "$OPTIMIZER" == muon ]]; then
    TRAINING_ARGS+=(
        --muon-momentum "$MUON_MOMENTUM"
        --muon-num-ns-steps "$MUON_NUM_NS_STEPS"
        --muon-scale-mode "$MUON_SCALE_MODE"
        --muon-extra-scale-factor "$MUON_EXTRA_SCALE_FACTOR"
        --muon-scalar-optimizer "$MUON_SCALAR_OPTIMIZER"
    )
    if [[ "$MUON_SPLIT_QKV" != true ]]; then
        TRAINING_ARGS+=(--muon-no-split-qkv)
    fi
    if [[ "$MUON_NESTEROV" == true ]]; then
        TRAINING_ARGS+=(--muon-nesterov)
    fi
else
    TRAINING_ARGS+=(
        --use-precision-aware-optimizer
        --main-params-dtype "$MAIN_PARAMS_DTYPE"
        --main-grads-dtype "$MAIN_GRADS_DTYPE"
        --exp-avg-dtype "$EXP_AVG_DTYPE"
        --exp-avg-sq-dtype "$EXP_AVG_SQ_DTYPE"
    )
fi
if [[ "$LR_DECAY_STYLE" == "WSD" ]]; then
    TRAINING_ARGS+=(
        --lr-wsd-decay-style "$LR_WSD_DECAY_STYLE"
        --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    )
else
    TRAINING_ARGS+=(--lr-decay-iters "$LR_DECAY_ITERS")
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
    --log-interval "$LOG_INTERVAL"
    --log-throughput
    --exit-signal-handler
    --ckpt-format torch_dist
)
if [[ -n "$VALID_DATA_PATH" ]]; then
    LOGGING_ARGS+=(--eval-iters "$EVAL_ITERS")
else
    LOGGING_ARGS+=(--eval-iters 0)
fi
if [[ -n "$LOAD_CHECKPOINT" ]]; then
    LOGGING_ARGS+=(
        --load "$LOAD_CHECKPOINT"
        --exit-on-missing-checkpoint
    )
fi
if [[ "$CHIMERA_CONTEXT_EXTENSION" == true ]]; then
    LOGGING_ARGS+=(
        --finetune
        --no-load-optim
        --no-load-rng
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
TRAIN_TOKENS=${TRAIN_TOKENS}
TOKENS_PER_ITER=${TOKENS_PER_ITER}
TRAIN_ITERS=${TRAIN_ITERS}
LR=${LR}
MIN_LR=${MIN_LR}
LR_DECAY_STYLE=${LR_DECAY_STYLE}
LR_DECAY_ITERS=${LR_DECAY_ITERS}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS}
LR_WSD_DECAY_STYLE=${LR_WSD_DECAY_STYLE}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS}
SAVE_INTERVAL=${SAVE_INTERVAL}
EVAL_INTERVAL=${EVAL_INTERVAL}
SEQ_LENGTH=${SEQ_LENGTH}
CONTEXT_PHASE=${CONTEXT_PHASE}
POSITION_EMBEDDING_TYPE=${POSITION_EMBEDDING_TYPE}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS}
ROTARY_SCALING_FACTOR=${ROTARY_SCALING_FACTOR}
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}
CHIMERA_CONTEXT_EXTENSION=${CHIMERA_CONTEXT_EXTENSION}
WEIGHT_DECAY=${WEIGHT_DECAY}
CLIP_GRAD=${CLIP_GRAD}
OPTIMIZER=${OPTIMIZER}
MUON_MOMENTUM=${MUON_MOMENTUM}
MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS}
MUON_SCALE_MODE=${MUON_SCALE_MODE}
MUON_EXTRA_SCALE_FACTOR=${MUON_EXTRA_SCALE_FACTOR}
MUON_SCALAR_OPTIMIZER=${MUON_SCALAR_OPTIMIZER}
MUON_SPLIT_QKV=${MUON_SPLIT_QKV}
MUON_NESTEROV=${MUON_NESTEROV}
ADAM_BETA1=${ADAM_BETA1}
ADAM_BETA2=${ADAM_BETA2}
ADAM_EPS=${ADAM_EPS}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
QK_LAYERNORM=${QK_LAYERNORM}
QK_CLIP=${QK_CLIP}
QK_CLIP_THRESHOLD=${QK_CLIP_THRESHOLD}
QK_CLIP_ALPHA=${QK_CLIP_ALPHA}
LOG_MAX_ATTENTION_LOGIT=${LOG_MAX_ATTENTION_LOGIT}
APPLY_ROPE_FUSION=${APPLY_ROPE_FUSION}
FUSED_LINEAR_CROSS_ENTROPY=${FUSED_LINEAR_CROSS_ENTROPY}
MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE}
MAIN_PARAMS_DTYPE=${MAIN_PARAMS_DTYPE}
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
echo "  Architecture:     layers=25 moe_layer_freq=[0]*2+[1]*23 hidden=2048 experts=32 topk=4 expert_ffn=2048 shared=0 qk_norm=$QK_LAYERNORM"
echo "  Router:           quantile_balancing bins=1000 ema=0 aux=0 bias_rate=0 scale=2.5 z_loss=1e-3"
echo "  Router logging:   inline interval=1 raw_expert_files=false"
echo "  Context/YaRN:     phase=$CONTEXT_PHASE max=$MAX_POSITION_EMBEDDINGS factor=$ROTARY_SCALING_FACTOR original=$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS extension=$CHIMERA_CONTEXT_EXTENSION"
echo "  Seq/batch/iters:  seq=$SEQ_LENGTH micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE iters=$TRAIN_ITERS"
echo "  LR schedule:      $LR_DECAY_STYLE peak=$LR min=$MIN_LR warmup=$LR_WARMUP_ITERS wsd_decay=$LR_WSD_DECAY_ITERS wsd_style=$LR_WSD_DECAY_STYLE"
if [[ "$OPTIMIZER" == muon ]]; then
    echo "  Optimizer:        Muon momentum=$MUON_MOMENTUM ns_steps=$MUON_NUM_NS_STEPS scale=$MUON_SCALE_MODE extra_scale=$MUON_EXTRA_SCALE_FACTOR split_qkv=$MUON_SPLIT_QKV nesterov=$MUON_NESTEROV scalar_optimizer=$MUON_SCALAR_OPTIMIZER state=fp32 wd=$WEIGHT_DECAY"
else
    echo "  Optimizer:        AdamW beta1=$ADAM_BETA1 beta2=$ADAM_BETA2 eps=$ADAM_EPS wd=$WEIGHT_DECAY main_params=$MAIN_PARAMS_DTYPE main_grads=$MAIN_GRADS_DTYPE exp_avg=$EXP_AVG_DTYPE exp_avg_sq=$EXP_AVG_SQ_DTYPE"
fi
if [[ "$CHIMERA_CONTEXT_EXTENSION" == true ]]; then
    echo "  Extension budget: target_tokens=$TRAIN_TOKENS actual_tokens=$((TRAIN_ITERS * TOKENS_PER_ITER))"
else
    echo "  Production budget: target_tokens=$TRAIN_TOKENS actual_tokens=$((TRAIN_ITERS * TOKENS_PER_ITER))"
fi
echo "  Attention:        backend=$ATTENTION_BACKEND qk_clip=$QK_CLIP threshold=$QK_CLIP_THRESHOLD alpha=$QK_CLIP_ALPHA log_max=$LOG_MAX_ATTENTION_LOGIT cuda_graph=TE:attn"
echo "  Fusions/clip:     rope=$APPLY_ROPE_FUSION linear_ce=$FUSED_LINEAR_CROSS_ENTROPY grad=$CLIP_GRAD"
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
