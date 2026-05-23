#!/bin/bash
set -eu
set -o pipefail

# train.sh: Production Training Engine for Gemma 3 CPT/SFT (1B/4B/12B)
# Optimized for single-node training on NVIDIA H200/H100/A100 GPUs.

echo "--- Gemma 3 Production Training Engine ---"

# ============================================================================
# CRITICAL: Force in-order kernel launch & dynamic memory allocation
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Argument Parsing
# ============================================================================
MODE="cpt"
MODEL_SIZE="4b"
MCORE_PATH=""
DATA_PATH=""
VALID_DATA_PATH=""
SAVE_PATH=""
ITERS=20000
LR=2e-5
WARMUP_ITERS=1000
GBS=32
MBS=1
SEQ_LEN=16384
WANDB_PROJECT=""
WANDB_EXP_NAME=""
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL=""
ATTENTION_BACKEND="flash"
RECOMPUTE_GRANULARITY="selective"
RECOMPUTE_METHOD=""
RECOMPUTE_NUM_LAYERS=""
FUSED_LINEAR_CROSS_ENTROPY=true
LOG_THROUGHPUT=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --model-size) MODEL_SIZE="$2"; shift 2 ;;
    --mcore-path) MCORE_PATH="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --valid-data-path) VALID_DATA_PATH="$2"; shift 2 ;;
    --save-path) SAVE_PATH="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
    --global-batch-size) GBS="$2"; shift 2 ;;
    --micro-batch-size) MBS="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-exp-name) WANDB_EXP_NAME="$2"; shift 2 ;;
    --tokenizer-type) TOKENIZER_TYPE="$2"; shift 2 ;;
    --tokenizer-model) TOKENIZER_MODEL="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --recompute-granularity) RECOMPUTE_GRANULARITY="$2"; shift 2 ;;
    --recompute-method) RECOMPUTE_METHOD="$2"; shift 2 ;;
    --recompute-num-layers) RECOMPUTE_NUM_LAYERS="$2"; shift 2 ;;
    --fused-linear-cross-entropy) FUSED_LINEAR_CROSS_ENTROPY=true; shift 1 ;;
    --log-throughput) LOG_THROUGHPUT=true; shift 1 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# ============================================================================
# Validation
# ============================================================================
[[ -z "$MCORE_PATH" ]] && echo "ERROR: --mcore-path is required." && exit 1
[[ -z "$DATA_PATH" ]]  && echo "ERROR: --data-path is required."  && exit 1

if [[ -z "$SAVE_PATH" ]]; then
  SAVE_PATH="/home/jovyan/data/checkpoints/gemma3-${MODEL_SIZE}-${MODE}"
  echo "INFO: No --save-path specified. Using default: $SAVE_PATH"
fi
mkdir -p "$SAVE_PATH"

if [[ -z "$TOKENIZER_MODEL" ]]; then
  TOKENIZER_MODEL="/home/jovyan/models/gemma-3-${MODEL_SIZE}-pt"
  echo "INFO: No --tokenizer-model specified. Using default: $TOKENIZER_MODEL"
fi

# ============================================================================
# Architecture Configuration
# ============================================================================
case $MODEL_SIZE in
    "1b")
        LAYERS=26; HIDDEN=1152; HEADS=4; GQA=1; KV_CH=256; FFN=6912
        WINDOW=512; VOCAB=262144; TP=1; PP=1
        ;;
    "4b")
        LAYERS=34; HIDDEN=2560; HEADS=8; GQA=4; KV_CH=256; FFN=10240
        WINDOW=1024; VOCAB=262208; TP=1; PP=1
        ;;
    "12b")
        LAYERS=48; HIDDEN=3840; HEADS=16; GQA=8; KV_CH=256; FFN=15360
        WINDOW=1024; VOCAB=262208; TP=2; PP=1
        ;;
    *) echo "Unsupported model size: $MODEL_SIZE"; exit 1 ;;
esac

# ============================================================================
# GPU Detection & Parallelism Safety
# ============================================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)

if [[ "$NUM_GPUS" -lt "$((TP * PP))" ]]; then
    echo "WARNING: Only $NUM_GPUS GPU(s) detected but TP=$TP PP=$PP requires $((TP * PP)). Overriding to TP=1 PP=1."
    TP=1; PP=1
fi

DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPUS --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6543"

# ============================================================================
# Model Architecture Args
# ============================================================================
MODEL_ARGS="
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --num-layers $LAYERS \
    --hidden-size $HIDDEN \
    --ffn-hidden-size $FFN \
    --num-attention-heads $HEADS \
    --group-query-attention \
    --num-query-groups $GQA \
    --kv-channels $KV_CH \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --window-size $WINDOW \
    --position-embedding-type rope \
    --no-position-embedding \
    --qk-layernorm \
    --normalization RMSNorm \
    --disable-bias-linear \
    --no-masked-softmax-fusion \
    --make-vocab-size-divisible-by 1 \
    --bf16 \
    --use-flash-attn \
    --attention-backend $ATTENTION_BACKEND \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type $TOKENIZER_TYPE \
    --tokenizer-model $TOKENIZER_MODEL
"

# ============================================================================
# Optimizer Args
# ============================================================================
OPTIM_ARGS="
    --lr $LR \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-iters $ITERS \
    --lr-warmup-iters $WARMUP_ITERS \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.01 \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-params-dtype fp16 \
    --main-grads-dtype bf16 \
    --grad-reduce-in-bf16 \
    --exp-avg-dtype fp16 \
    --exp-avg-sq-dtype fp16
"

# ============================================================================
# Training & Recompute Args
# ============================================================================
RECOMPUTE_ARGS=""
if [[ "$RECOMPUTE_GRANULARITY" != "none" ]]; then
    RECOMPUTE_ARGS="--recompute-granularity $RECOMPUTE_GRANULARITY"
fi
if [[ -n "$RECOMPUTE_METHOD" ]]; then
    RECOMPUTE_ARGS="$RECOMPUTE_ARGS --recompute-method $RECOMPUTE_METHOD"
fi
if [[ -n "$RECOMPUTE_NUM_LAYERS" ]]; then
    RECOMPUTE_ARGS="$RECOMPUTE_ARGS --recompute-num-layers $RECOMPUTE_NUM_LAYERS"
fi

TRAIN_ARGS=" 
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-iters $ITERS \
    $RECOMPUTE_ARGS \
    --num-workers 4 \
    --manual-gc \
    --manual-gc-interval 5 \
    --overlap-grad-reduce \
    --overlap-param-gather
"

# ============================================================================
# Logging Args
# ============================================================================
LOG_ARGS="
    --load $MCORE_PATH \
    --save $SAVE_PATH \
    --save-interval 1000 \
    --log-interval 10 \
    --eval-interval 500 \
    --eval-iters 10 \
    --no-load-optim \
    --no-load-rng \
    --finetune
"

# ============================================================================
# Data Args
# ============================================================================
if [[ -n "$VALID_DATA_PATH" ]]; then
    DATA_ARGS="--train-data-path 1.0 $DATA_PATH --valid-data-path 1.0 $VALID_DATA_PATH"
else
    DATA_ARGS="--data-path 1.0 $DATA_PATH"
fi

if [[ "$MODE" == "sft" ]]; then
    DATA_ARGS="$DATA_ARGS --is-instruction-dataset"
    TRAIN_ARGS=" $TRAIN_ARGS --reset-position-ids --reset-attention-mask --eod-mask-loss"
fi

# ============================================================================
# WandB & Extra Args
# ============================================================================
WANDB_ARGS=""
if [[ -n "$WANDB_PROJECT" ]]; then
    WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-exp-name ${WANDB_EXP_NAME:-gemma3-${MODEL_SIZE}-${MODE}}"
fi

EXTRA_ARGS=""
if [ "$FUSED_LINEAR_CROSS_ENTROPY" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --fused-linear-cross-entropy"
fi
if [ "$LOG_THROUGHPUT" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --log-throughput"
fi

# ============================================================================
# Launch
# ============================================================================
echo "================================================================="
echo "  Mode:         $MODE"
echo "  Model:        Gemma 3 $MODEL_SIZE"
echo "  GPUs:         $NUM_GPUS (TP=$TP, PP=$PP)"
echo "  Seq Length:    $SEQ_LEN"
echo "  GBS:          $GBS (MBS=$MBS, Accum Steps=$((GBS / MBS / NUM_GPUS)))"
echo "  Recompute:    $RECOMPUTE_GRANULARITY ($RECOMPUTE_METHOD $RECOMPUTE_NUM_LAYERS)"
echo "  LR:           $LR (warmup=$WARMUP_ITERS, decay=$ITERS iters)"
echo "  Checkpoints:  $SAVE_PATH"
echo "  CCE loss:     $FUSED_LINEAR_CROSS_ENTROPY"
echo "================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun $DISTRIBUTED_ARGS \
    "$SCRIPT_DIR/utils/pretrain_gemma3_mcore.py" \
    $MODEL_ARGS \
    $OPTIM_ARGS \
    $TRAIN_ARGS \
    $LOG_ARGS \
    $DATA_ARGS \
    $WANDB_ARGS \
    $EXTRA_ARGS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP
