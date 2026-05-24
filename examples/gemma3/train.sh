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
ITERS=0
LR=0
MIN_LR=5e-7
WARMUP_ITERS=0
DECAY_ITERS=0
GBS=64
MBS=1
SEQ_LEN=16384
WANDB_PROJECT="AUTO"
WANDB_EXP_NAME=""
TENSORBOARD_DIR=""
TOKENIZER_TYPE=""
TOKENIZER_MODEL=""
SFT_PROMPT_FORMAT="gemma3"
ATTENTION_BACKEND="flash"
RECOMPUTE_GRANULARITY="full"
RECOMPUTE_METHOD="uniform"
RECOMPUTE_NUM_LAYERS="2"
FUSED_LINEAR_CROSS_ENTROPY=true
LINEAR_CE_FILTER_E_GRAD=false
LINEAR_CE_FILTER_C_GRAD=false
LINEAR_CE_FILTER_EPS="0.0"
LOG_THROUGHPUT=true
SAVE_INTERVAL=0
EVAL_INTERVAL=0
LOG_INTERVAL=1
EPOCHS=1.0
WARMUP_PRCT=5
DECAY_PRCT=90
PACK_SAMPLES=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --pack-samples) PACK_SAMPLES=true; shift 1 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --warmup-prct) WARMUP_PRCT="$2"; shift 2 ;;
    --decay-prct) DECAY_PRCT="$2"; shift 2 ;;
    --model-size) MODEL_SIZE="$2"; shift 2 ;;
    --mcore-path) MCORE_PATH="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --valid-data-path) VALID_DATA_PATH="$2"; shift 2 ;;
    --save-path) SAVE_PATH="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --min-lr) MIN_LR="$2"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
    --lr-decay-iters) DECAY_ITERS="$2"; shift 2 ;;
    --global-batch-size) GBS="$2"; shift 2 ;;
    --micro-batch-size) MBS="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-exp-name) WANDB_EXP_NAME="$2"; shift 2 ;;
    --tensorboard-dir) TENSORBOARD_DIR="$2"; shift 2 ;;
    --tokenizer-type) TOKENIZER_TYPE="$2"; shift 2 ;;
    --tokenizer-model) TOKENIZER_MODEL="$2"; shift 2 ;;
    --sft-prompt-format) SFT_PROMPT_FORMAT="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --recompute-granularity) RECOMPUTE_GRANULARITY="$2"; shift 2 ;;
    --recompute-method) RECOMPUTE_METHOD="$2"; shift 2 ;;
    --recompute-num-layers) RECOMPUTE_NUM_LAYERS="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --fused-linear-cross-entropy) FUSED_LINEAR_CROSS_ENTROPY=true; shift 1 ;;
    --no-fused-linear-cross-entropy) FUSED_LINEAR_CROSS_ENTROPY=false; shift 1 ;;
    --linear-ce-filter-e-grad) LINEAR_CE_FILTER_E_GRAD=true; shift 1 ;;
    --linear-ce-filter-c-grad) LINEAR_CE_FILTER_C_GRAD=true; shift 1 ;;
    --no-linear-ce-filter-e-grad) LINEAR_CE_FILTER_E_GRAD=false; shift 1 ;;
    --no-linear-ce-filter-c-grad) LINEAR_CE_FILTER_C_GRAD=false; shift 1 ;;
    --linear-ce-filter-eps) LINEAR_CE_FILTER_EPS="$2"; shift 2 ;;
    --log-throughput) LOG_THROUGHPUT=true; shift 1 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# ============================================================================
# Validation & Mode-Aware Defaults
# ============================================================================

# 0. Tensorboard IST Timestamp Logic (Unlocks WandB metrics)
if [[ -z "$TENSORBOARD_DIR" ]]; then
    IST_DATE=$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)
    TENSORBOARD_DIR="/home/jovyan/logs/tb/${MODE}_${IST_DATE}"
fi
mkdir -p "$TENSORBOARD_DIR"

# 1. Default Tokenizer Logic
if [[ -z "$TOKENIZER_TYPE" ]]; then
    if [[ "$MODE" == "cpt" ]]; then
        TOKENIZER_TYPE="HuggingFaceTokenizer"
    elif [[ "$MODE" == "sft" ]]; then
        TOKENIZER_TYPE="SFTTokenizer"
    fi
fi

# 2. Default MCORE_PATH logic
if [[ -z "$MCORE_PATH" ]]; then
  if [[ "$MODE" == "cpt" ]]; then
    MCORE_PATH="/home/jovyan/models/gemma-3-${MODEL_SIZE}-pt-mcore"
  elif [[ "$MODE" == "sft" ]]; then
    MCORE_PATH="/home/jovyan/data/checkpoints/gemma3-${MODEL_SIZE}-cpt"
  fi
  echo "INFO: No --mcore-path specified. Using default for $MODE mode: $MCORE_PATH"
fi

# 3. Default DATA_PATH logic
if [[ -z "$DATA_PATH" ]]; then
  if [[ "$MODE" == "cpt" ]]; then
    DATA_PATH="/home/jovyan/data/pubmed_train_text_document"
    VALID_DATA_PATH="/home/jovyan/data/pubmed_val_text_document"
  elif [[ "$MODE" == "sft" ]]; then
    DATA_PATH="/home/jovyan/data/sft_train.jsonl"
    VALID_DATA_PATH="/home/jovyan/data/sft_val.jsonl"
  fi
  echo "INFO: No --data-path specified. Using default for $MODE mode: $DATA_PATH"
fi

# 4. Default Hyperparameter Logic (Branching)
if [[ "$MODE" == "sft" ]]; then
    # SFT Profile: 18,772 samples, GBS 64, dynamic Epoch calculation
    if [ "$PACK_SAMPLES" = true ]; then
        # Dynamically calculate precise ITERS using SFT token scanner
        if [[ $ITERS -eq 0 ]]; then
            echo "Calculating exact SFT training iterations dynamically..."
            ITERS=$(python3 examples/gemma3/utils/get_sft_tokens.py "$DATA_PATH" "$TOKENIZER_MODEL" "$EPOCHS" "$GBS" "$SEQ_LEN")
            echo "Calculated training iterations: $ITERS"
        fi
        
        # Scale lr scheduler and evaluation intervals dynamically by percentages
        [[ $WARMUP_ITERS -eq 0 ]] && WARMUP_ITERS=$(( ITERS * WARMUP_PRCT / 100 ))
        [[ $WARMUP_ITERS -eq 0 ]] && WARMUP_ITERS=1 # At least 1 step
        
        [[ $DECAY_ITERS -eq 0 ]] && DECAY_ITERS=$(( ITERS * DECAY_PRCT / 100 ))
        
        [[ $SAVE_INTERVAL -eq 0 ]] && SAVE_INTERVAL=$(( ITERS * 10 / 100 )) # Save every 10%
        [[ $SAVE_INTERVAL -eq 0 ]] && SAVE_INTERVAL=5
        
        [[ $EVAL_INTERVAL -eq 0 ]] && EVAL_INTERVAL=$(( ITERS * 5 / 100 )) # Eval every 5%
        [[ $EVAL_INTERVAL -eq 0 ]] && EVAL_INTERVAL=2
    else
        # Unpacked baseline: scale by EPOCHS
        # Standard 1 Epoch base = 294 steps. Scale linearly with requested epochs.
        if [[ $ITERS -eq 0 ]]; then
            ITERS=$(python3 -c "import math; print(math.ceil(294 * $EPOCHS))")
        fi
        [[ $WARMUP_ITERS -eq 0 ]] && WARMUP_ITERS=$(( ITERS * WARMUP_PRCT / 100 ))
        [[ $WARMUP_ITERS -eq 0 ]] && WARMUP_ITERS=1
        [[ $DECAY_ITERS -eq 0 ]] && DECAY_ITERS=$(( ITERS * DECAY_PRCT / 100 ))
        [[ $SAVE_INTERVAL -eq 0 ]] && SAVE_INTERVAL=$(( ITERS * 10 / 100 ))
        [[ $SAVE_INTERVAL -eq 0 ]] && SAVE_INTERVAL=5
        [[ $EVAL_INTERVAL -eq 0 ]] && EVAL_INTERVAL=$(( ITERS * 5 / 100 ))
        [[ $EVAL_INTERVAL -eq 0 ]] && EVAL_INTERVAL=2
    fi
    [[ $LR == "0" ]] && LR=1e-6
    if [[ "$WANDB_PROJECT" == "AUTO" ]]; then WANDB_PROJECT="gemma3-medical-sft-reasoning"; fi
    [[ $SEQ_LEN -eq 16384 ]] && SEQ_LEN=8192 
else
    # CPT Profile: 500M token budget
    [[ $ITERS -eq 0 ]] && ITERS=476
    [[ $WARMUP_ITERS -eq 0 ]] && WARMUP_ITERS=9
    [[ $DECAY_ITERS -eq 0 ]] && DECAY_ITERS=428
    [[ $SAVE_INTERVAL -eq 0 ]] && SAVE_INTERVAL=48
    [[ $EVAL_INTERVAL -eq 0 ]] && EVAL_INTERVAL=24
    [[ $LR == "0" ]] && LR=5e-6
    if [[ "$WANDB_PROJECT" == "AUTO" ]]; then WANDB_PROJECT="gemma3-medical-cpt-prod"; fi
fi

# Final Cleanup for WandB (if explicitly disabled)
if [[ "$WANDB_PROJECT" == "NONE" || -z "$WANDB_PROJECT" ]]; then
    WANDB_PROJECT=""
fi

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

DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPUS --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6789"

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
    --tokenizer-model $TOKENIZER_MODEL \
    --sft-tokenizer-prompt-format $SFT_PROMPT_FORMAT
"

# ============================================================================
# Optimizer Args
# ============================================================================
OPTIM_ARGS="
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --lr-decay-iters $DECAY_ITERS \
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
    --tensorboard-dir $TENSORBOARD_DIR \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --log-interval $LOG_INTERVAL \
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
    DATA_ARGS="$DATA_ARGS --sft"
    TRAIN_ARGS=" $TRAIN_ARGS --eod-mask-loss --no-create-attention-mask-in-dataloader"
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
    if [ "$LINEAR_CE_FILTER_E_GRAD" = false ]; then
        EXTRA_ARGS="$EXTRA_ARGS --no-linear-ce-filter-e-grad"
    fi
    if [ "$LINEAR_CE_FILTER_C_GRAD" = false ]; then
        EXTRA_ARGS="$EXTRA_ARGS --no-linear-ce-filter-c-grad"
    fi
    if [ "$LINEAR_CE_FILTER_EPS" != "auto" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --linear-ce-filter-eps $LINEAR_CE_FILTER_EPS"
    fi
fi
if [ "$LOG_THROUGHPUT" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --log-throughput"
fi
if [ "$PACK_SAMPLES" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --pack-samples"
fi

# ============================================================================
# Launch
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_LAUNCH_ARGS=()
if [[ "$MODE" == "sft" ]]; then
    TEMPLATE_PATH="$SCRIPT_DIR/utils/gemma3_chat_template.jinja"
    if [[ -f "$TEMPLATE_PATH" ]]; then
        # Read template and strip newlines to safely pass as single string argument
        CHAT_TEMPLATE=$(cat "$TEMPLATE_PATH" | tr -d '\n')
        EXTRA_LAUNCH_ARGS+=("--chat-template" "$CHAT_TEMPLATE")
    else
        echo "WARNING: Chat template not found at $TEMPLATE_PATH"
    fi
fi

echo "================================================================="
echo "  Mode:         $MODE"
echo "  Model:        Gemma 3 $MODEL_SIZE"
echo "  GPUs:         $NUM_GPUS (TP=$TP, PP=$PP)"
echo "  Seq Length:    $SEQ_LEN"
echo "  GBS:          $GBS (MBS=$MBS, Accum Steps=$((GBS / MBS / NUM_GPUS)))"
echo "  Recompute:    $RECOMPUTE_GRANULARITY ($RECOMPUTE_METHOD $RECOMPUTE_NUM_LAYERS)"
echo "  LR:           $LR (warmup=$WARMUP_ITERS, decay=$DECAY_ITERS iters)"
echo "  Checkpoints:  $SAVE_PATH"
echo "  Tensorboard:  $TENSORBOARD_DIR"
echo "  CCE loss:     $FUSED_LINEAR_CROSS_ENTROPY"
echo "  Tokenizer:    $TOKENIZER_TYPE ($SFT_PROMPT_FORMAT)"
echo "  WandB:        ${WANDB_PROJECT:-DISABLED}"
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
    --pipeline-model-parallel-size $PP \
    "${EXTRA_LAUNCH_ARGS[@]}"
