#!/bin/bash
set -euo pipefail

# Required user inputs.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data/simpo/overfit.jsonl}"
VALID_DATA_PATH="${VALID_DATA_PATH:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/datasets/megadata/hf_models/chimera-10b}"
MCORE_PATH="${MCORE_PATH:-/datasets/megadata/chimera_sft_runs/latest/checkpoints}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_simpo_runs}"
INTRA_DOC_MASKING="${INTRA_DOC_MASKING:-false}"

[[ "$INTRA_DOC_MASKING" == false ]] || {
    echo "SimPO requires INTRA_DOC_MASKING=false; chosen/rejected sequences are isolated by cu_seqlens." >&2
    exit 1
}

# Distributed launch settings.
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29593}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"

# Training settings.
CONTEXT_PHASE="${CONTEXT_PHASE:-128k}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
source "$SCRIPT_DIR/context_phase.sh"
source "$SCRIPT_DIR/schedule_helpers.sh"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-}"
LR="${LR:-8e-7}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.10}"
LR_DECAY_FRACTION="${LR_DECAY_FRACTION:-1.0}"
LR_WSD_DECAY_FRACTION="${LR_WSD_DECAY_FRACTION:-0.10}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-linear}"
SAVE_INTERVAL_FRACTION="${SAVE_INTERVAL_FRACTION:-1.0}"
EVAL_INTERVAL_FRACTION="${EVAL_INTERVAL_FRACTION:-1.0}"
EVAL_ITERS="${EVAL_ITERS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PREPARE_WORKERS="${PREPARE_WORKERS:-32}"
SAVE_WEIGHTS_ONLY="${SAVE_WEIGHTS_ONLY:-false}"
PACK_SAMPLES="${PACK_SAMPLES:-true}"
PACK_METADATA_PATH="${PACK_METADATA_PATH:-${DATA_PATH}.chimera_simpo_packing}"
VALID_PACK_METADATA_PATH="${VALID_PACK_METADATA_PATH:-${VALID_DATA_PATH:+${VALID_DATA_PATH}.chimera_simpo_packing}}"
OPTIMIZER="${OPTIMIZER:-adam}"
MUON_NUM_NS_STEPS="${MUON_NUM_NS_STEPS:-6}"
MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"
FUSED_LINEAR_CROSS_ENTROPY="${FUSED_LINEAR_CROSS_ENTROPY:-true}"

[[ "$OPTIMIZER" == adam || "$OPTIMIZER" == muon ]] || {
    echo "Unsupported OPTIMIZER=$OPTIMIZER; expected adam or muon" >&2
    exit 1
}
[[ "$LR_DECAY_STYLE" == cosine || "$LR_DECAY_STYLE" == WSD ]] || {
    echo "Unsupported LR_DECAY_STYLE=$LR_DECAY_STYLE; expected cosine or WSD" >&2
    exit 1
}

# SimPO settings.
SIMPO_BETA="${SIMPO_BETA:-2.5}"
SIMPO_GAMMA="${SIMPO_GAMMA:-0.55}"
SIMPO_LOSS_TYPE="${SIMPO_LOSS_TYPE:-sigmoid}"
SIMPO_SFT_WEIGHT="${SIMPO_SFT_WEIGHT:-0.0}"

RUN_STAMP="${RUN_STAMP:-$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUNS_ROOT}/${RUN_STAMP}"
SAVE_PATH="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATA_CACHE_PATH="${RUN_DIR}/data_cache"
LOG_DIR="${RUN_DIR}/logs"

[[ -f "$DATA_PATH" ]] || { echo "Missing SimPO JSONL: $DATA_PATH"; exit 1; }
if [[ -n "$VALID_DATA_PATH" ]]; then
    [[ -f "$VALID_DATA_PATH" ]] || { echo "Missing validation SimPO JSONL: $VALID_DATA_PATH"; exit 1; }
fi
[[ -d "$TOKENIZER_MODEL" || -f "$TOKENIZER_MODEL" ]] || { echo "Missing tokenizer model: $TOKENIZER_MODEL"; exit 1; }
[[ -d "$MCORE_PATH" ]] || { echo "Missing source MCore checkpoint: $MCORE_PATH"; exit 1; }
if [[ -d "$MCORE_PATH" ]]; then
    SOURCE_RUN_CONFIG="$MCORE_PATH/run_config.yaml"
    if [[ ! -f "$SOURCE_RUN_CONFIG" ]]; then
        SOURCE_RUN_CONFIG=$(find "$MCORE_PATH" -name run_config.yaml -type f -print -quit)
    fi
    [[ -n "$SOURCE_RUN_CONFIG" && -f "$SOURCE_RUN_CONFIG" ]] || {
        echo "Missing checkpoint run_config.yaml under $MCORE_PATH"; exit 1;
    }
    python3 "$SCRIPT_DIR/architecture_contract.py" validate-run-config "$SOURCE_RUN_CONFIG" \
        --profile full --context-phase "$CONTEXT_PHASE"
fi

read -r DATASET_SAMPLES < <(wc -l < "$DATA_PATH")
(( DATASET_SAMPLES > 0 )) || { echo "SimPO JSONL is empty: $DATA_PATH"; exit 1; }
SCHEDULE_SAMPLES=$DATASET_SAMPLES
if [[ "$PACK_SAMPLES" == true ]]; then
    if [[ ! -f "$PACK_METADATA_PATH/metadata.json" || ! -f "$PACK_METADATA_PATH/lengths.npy" || ! -f "$PACK_METADATA_PATH/row_offsets.npy" ]]; then
        echo "Packing metadata missing. Automatically generating metadata in: $PACK_METADATA_PATH"
        python3 "$SCRIPT_DIR/prepare_chat_data.py" \
            --mode simpo \
            --input "$DATA_PATH" \
            --output "$PACK_METADATA_PATH" \
            --tokenizer-model "$TOKENIZER_MODEL" \
            --workers "$PREPARE_WORKERS"
    fi
    SCHEDULE_SAMPLES=$(python3 "$SCRIPT_DIR/count_chat_packs.py" \
        --metadata "$PACK_METADATA_PATH" --mode simpo --sequence-length "$SEQ_LENGTH")
fi
if [[ "$PACK_SAMPLES" == true && -n "$VALID_DATA_PATH" ]]; then
    if [[ ! -f "$VALID_PACK_METADATA_PATH/metadata.json" || ! -f "$VALID_PACK_METADATA_PATH/lengths.npy" || ! -f "$VALID_PACK_METADATA_PATH/row_offsets.npy" ]]; then
        echo "Validation packing metadata missing. Automatically generating: $VALID_PACK_METADATA_PATH"
        python3 "$SCRIPT_DIR/prepare_chat_data.py" \
            --mode simpo \
            --input "$VALID_DATA_PATH" \
            --output "$VALID_PACK_METADATA_PATH" \
            --tokenizer-model "$TOKENIZER_MODEL" \
            --workers "$PREPARE_WORKERS"
    fi
fi
if [[ -z "$TRAIN_ITERS" ]]; then
    TRAIN_ITERS=$(( (SCHEDULE_SAMPLES * TRAIN_EPOCHS + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
fi
chimera_resolve_schedule

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR" "$DATA_CACHE_PATH" "$LOG_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

NUM_LAYERS="${NUM_LAYERS:-25}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-8192}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"
KV_CHANNELS="${KV_CHANNELS:-256}"
NUM_EXPERTS="${NUM_EXPERTS:-32}"
MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*2+[1]*23}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-2048}"

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --group-query-attention
    --num-query-groups "$NUM_QUERY_GROUPS"
    --kv-channels "$KV_CHANNELS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type "$POSITION_EMBEDDING_TYPE"
    --apply-rope-fusion
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
    --tokenizer-type SFTTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --sft-tokenizer-prompt-format chimera
    --fused-residual-rmsnorm
)

MOE_ARGS=(
    --num-experts "$NUM_EXPERTS"
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-topk "$MOE_ROUTER_TOPK"
    --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE"
    --moe-router-load-balancing-type "${MOE_ROUTER_LOAD_BALANCING_TYPE:-none}"
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.0}"
    --moe-qb-num-bins "${MOE_QB_NUM_BINS:-1000}"
    --moe-qb-ema-decay "${MOE_QB_EMA_DECAY:-0.0}"
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate "${MOE_ROUTER_BIAS_UPDATE_RATE:-0.0}"
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-z-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-fusion
    --moe-per-layer-logging
)

DATA_ARGS=(
    --train-data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --num-workers "$NUM_WORKERS"
    --sft
    --simpo
    --simpo-beta "$SIMPO_BETA"
    --simpo-gamma "$SIMPO_GAMMA"
    --simpo-loss-type "$SIMPO_LOSS_TYPE"
    --simpo-sft-weight "$SIMPO_SFT_WEIGHT"
    --eod-mask-loss
    --no-create-attention-mask-in-dataloader
)
if [[ -n "$VALID_DATA_PATH" ]]; then
    DATA_ARGS+=(--valid-data-path "$VALID_DATA_PATH")
fi
if [[ "$PACK_SAMPLES" == true ]]; then
    DATA_ARGS+=(--pack-samples --pack-metadata-path "$PACK_METADATA_PATH")
    if [[ -n "$VALID_PACK_METADATA_PATH" ]]; then
        DATA_ARGS+=(--valid-pack-metadata-path "$VALID_PACK_METADATA_PATH")
    fi
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
    --attention-softmax-in-fp32
    --manual-gc
    --manual-gc-interval 1000
    --use-distributed-optimizer
    --overlap-grad-reduce
)
if [[ "$LR_DECAY_STYLE" == WSD ]]; then
    TRAINING_ARGS+=(--lr-wsd-decay-style "$LR_WSD_DECAY_STYLE" --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS")
else
    TRAINING_ARGS+=(--lr-decay-iters "$LR_DECAY_ITERS")
fi

TRAINING_ARGS+=(--optimizer "$OPTIMIZER")

if [[ "$OPTIMIZER" == muon ]]; then
    TRAINING_ARGS+=(--muon-num-ns-steps "$MUON_NUM_NS_STEPS")
else
    TRAINING_ARGS+=(
        --use-precision-aware-optimizer
        --main-params-dtype "$MAIN_PARAMS_DTYPE"
        --main-grads-dtype "$MAIN_GRADS_DTYPE"
        --exp-avg-dtype "$EXP_AVG_DTYPE"
        --exp-avg-sq-dtype "$EXP_AVG_SQ_DTYPE"
    )
fi

if [[ "$FUSED_LINEAR_CROSS_ENTROPY" == true ]]; then
    TRAINING_ARGS+=(--fused-linear-cross-entropy)
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
    --eval-iters "$([[ -n "$VALID_DATA_PATH" ]] && echo "$EVAL_ITERS" || echo 0)"
    --log-interval "$LOG_INTERVAL"
    --log-throughput
    --exit-signal-handler
    --ckpt-format torch_dist
)
if [[ -n "${MCORE_PATH:-}" && -d "$MCORE_PATH" ]]; then
    LOGGING_ARGS+=(
        --load "$MCORE_PATH"
        --finetune
        --no-load-optim
        --no-load-rng
    )
fi
if [[ "$SAVE_WEIGHTS_ONLY" == true ]]; then
    LOGGING_ARGS+=(--no-save-optim --no-save-rng)
fi

if [[ "$NODE_RANK" == 0 ]]; then
cat > "${RUN_DIR}/run_paths.env" <<EOF
DATA_PATH=${DATA_PATH}
VALID_DATA_PATH=${VALID_DATA_PATH}
TOKENIZER_MODEL=${TOKENIZER_MODEL}
MCORE_PATH=${MCORE_PATH}
RUN_DIR=${RUN_DIR}
SAVE_PATH=${SAVE_PATH}
TP_SIZE=${TP_SIZE}
PP_SIZE=${PP_SIZE}
EP_SIZE=${EP_SIZE}
CP_SIZE=${CP_SIZE}
DATASET_SAMPLES=${DATASET_SAMPLES}
SCHEDULE_SAMPLES=${SCHEDULE_SAMPLES}
PACK_SAMPLES=${PACK_SAMPLES}
PACK_METADATA_PATH=${PACK_METADATA_PATH}
VALID_PACK_METADATA_PATH=${VALID_PACK_METADATA_PATH}
MAIN_PARAMS_DTYPE=${MAIN_PARAMS_DTYPE}
MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE}
EXP_AVG_DTYPE=${EXP_AVG_DTYPE}
EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE}
TRAIN_EPOCHS=${TRAIN_EPOCHS}
TRAIN_ITERS=${TRAIN_ITERS}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
SEQ_LENGTH=${SEQ_LENGTH}
CONTEXT_PHASE=${CONTEXT_PHASE}
POSITION_EMBEDDING_TYPE=${POSITION_EMBEDDING_TYPE}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS}
ROTARY_SCALING_FACTOR=${ROTARY_SCALING_FACTOR}
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}
EOF

    cp "$0" "${RUN_DIR}/simpo.sh"
fi

echo "Running Chimera SimPO"
echo "  Run dir:          $RUN_DIR"
echo "  Data JSONL:       $DATA_PATH"
echo "  Load checkpoint:  $MCORE_PATH"
echo "  Tokenizer:        $TOKENIZER_MODEL"
echo "  Parallelism:      TP=$TP_SIZE PP=$PP_SIZE EP=$EP_SIZE ETP=1 CP=$CP_SIZE"
echo "  Architecture:     layers=25 moe_layer_freq=[0]*2+[1]*23 hidden=2048 ffn=8192 experts=32 topk=4 expert_ffn=2048 shared=0 qk_norm=true"
echo "  Router:           ${MOE_ROUTER_LOAD_BALANCING_TYPE:-none} bins=${MOE_QB_NUM_BINS:-1000} ema=${MOE_QB_EMA_DECAY:-0.0} aux=${MOE_AUX_LOSS_COEFF:-0.0} bias_rate=${MOE_ROUTER_BIAS_UPDATE_RATE:-0.0} scale=2.5 z_loss=0.001"
echo "  Data schedule:    rows=$DATASET_SAMPLES samples=$SCHEDULE_SAMPLES epochs=$TRAIN_EPOCHS iters=$TRAIN_ITERS"
echo "  Context/YaRN:     phase=$CONTEXT_PHASE max=$MAX_POSITION_EMBEDDINGS factor=$ROTARY_SCALING_FACTOR original=$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"
echo "  Seq/batch:        seq=$SEQ_LENGTH micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE"
echo "  Optimizer:        type=$OPTIMIZER lr=$LR min_lr=$MIN_LR warmup=$LR_WARMUP_ITERS wd=$WEIGHT_DECAY params=$MAIN_PARAMS_DTYPE grads=$MAIN_GRADS_DTYPE moments=$EXP_AVG_DTYPE,$EXP_AVG_SQ_DTYPE"
echo "  SimPO:            beta=$SIMPO_BETA gamma=$SIMPO_GAMMA loss=$SIMPO_LOSS_TYPE sft_weight=$SIMPO_SFT_WEIGHT"
echo "  Packing:          $PACK_SAMPLES"
echo "  Intra-doc mask:   false (chosen/rejected isolation uses packed-sequence boundaries)"

exec python3 -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    examples/chimera/pretrain_chimera.py \
    --chimera-expert-tp-size 1 \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" > >(tee -a "${LOG_DIR}/simpo_node_${NODE_RANK}.log") 2>&1
