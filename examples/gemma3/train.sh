#!/bin/bash
set -eu
set -o pipefail

# train.sh: Production Training Engine for Gemma 3 CPT/SFT/SimPO (1B/4B/12B)
# Optimized for single-node training on NVIDIA H200/H100/A100 GPUs.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 1 — IDENTITY        (must pass for every run)
#   --mode          cpt | sft | simpo
#   --model-size    1b  | 4b  | 12b
#   --mcore-path    checkpoint to load from
#   --data-path     training data (Megatron binary prefix for CPT, JSONL for SFT/SimPO)
#
# TIER 2 — TRAINING BUDGET (one arg per mode; --iters overrides everything)
#   CPT:       --token-budget   tokens to train on  (default: 500M)
#   SFT/SimPO: --epochs         epochs to train for (default: 1.0)
#   All:       --iters          hard step count override (skips auto-calculation)
#
# TIER 3 — LEARNING RATE   (auto-set per mode, overridable)
#   --lr            CPT: 1e-5 | SFT: 5e-6 | SimPO: 1e-6
#   --min-lr        default: lr × 0.1
#   --warmup-prct   CPT: 2% | SFT/SimPO: 5%
#   --decay-prct    all: 90%
#
# TIER 4 — BATCH & SEQUENCE (auto per mode, overridable)
#   --global-batch-size   CPT: 64 | SFT/SimPO: 32
#   --micro-batch-size    all: 1
#   --seq-len             all: 8192
#
# TIER 5 — ALGORITHM PARAMS (SimPO only)
#   --simpo-beta, --simpo-gamma, --simpo-loss-type, --simpo-sft-weight
#
# TIER 6 — INFRA            (expert overrides, all have sensible defaults)
#   --valid-data-path     if not passed → no validation at all
#   --save-path, --save-interval, --eval-interval, --log-interval
#   --wandb-project, --wandb-exp-name, --tensorboard-dir
#   --tp-size, --nnodes, --node-rank, --master-addr, --master-port
#   --recompute-granularity, --recompute-method, --recompute-num-layers
#   --pack-factor         expert cap on samples per packed step (default: unlimited)
#   --split               CPT expert override (default: 100,0,0)
#   --tokenizer-type, --tokenizer-model, --sft-prompt-format
#   --weight-decay, --attention-backend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "--- Gemma 3 Production Training Engine ---"

# ============================================================================
# CRITICAL: Force in-order kernel launch & dynamic memory allocation
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# TIER 1 — IDENTITY
# ============================================================================
MODE="cpt"
MODEL_SIZE="4b"
MCORE_PATH=""
DATA_PATH=""

# ============================================================================
# TIER 2 — TRAINING BUDGET
# ============================================================================
TOKEN_BUDGET=500000000   # CPT: train until this many tokens seen
EPOCHS=1.0               # SFT/SimPO: number of epochs
ITERS=0                  # Hard override: if > 0, skips all auto-calculation

# ============================================================================
# TIER 3 — LEARNING RATE  (0 = auto-set per mode)
# ============================================================================
LR=0
MIN_LR=0
WARMUP_PRCT="auto"       # auto → CPT: 2%, SFT/SimPO: 5%
DECAY_PRCT=90

# ============================================================================
# TIER 4 — BATCH & SEQUENCE  (0 = auto-set per mode)
# ============================================================================
GBS=0                    # auto → CPT: 64, SFT/SimPO: 32
MBS=1
SEQ_LEN=8192

# ============================================================================
# TIER 5 — ALGORITHM PARAMS  (SimPO only)
# ============================================================================
SIMPO_BETA="2.0"
SIMPO_GAMMA="0.5"
SIMPO_LOSS_TYPE="sigmoid"
SIMPO_SFT_WEIGHT="0.0"

# ============================================================================
# TIER 6 — INFRA
# ============================================================================
VALID_DATA_PATH=""        # if not passed → no validation
SAVE_PATH=""
SAVE_INTERVAL=0
EVAL_INTERVAL=0
LOG_INTERVAL=1
EVAL_ITERS=2
WANDB_PROJECT="AUTO"
WANDB_EXP_NAME=""
TENSORBOARD_DIR=""
TOKENIZER_TYPE=""
TOKENIZER_MODEL=""
SFT_PROMPT_FORMAT="gemma3"
ATTENTION_BACKEND="flash"
RECOMPUTE_GRANULARITY="auto"
RECOMPUTE_METHOD="auto"
RECOMPUTE_NUM_LAYERS="auto"
FUSED_LINEAR_CROSS_ENTROPY=true
LINEAR_CE_FILTER_E_GRAD=false
LINEAR_CE_FILTER_C_GRAD=false
LINEAR_CE_FILTER_EPS="0.0"
LOG_THROUGHPUT=true
PACK_FACTOR=""            # expert: cap samples per packed step (default: unlimited)
SPLIT="auto"              # CPT expert override; default forced to 100,0,0
WEIGHT_DECAY=0.1
TP_OVERRIDE=0

# Multi-node
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=6789

# ============================================================================
# Argument Parsing
# ============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    # TIER 1
    --mode)               MODE="$2";         shift 2 ;;
    --model-size)         MODEL_SIZE="$2";   shift 2 ;;
    --mcore-path)         MCORE_PATH="$2";   shift 2 ;;
    --data-path)          DATA_PATH="$2";    shift 2 ;;
    # TIER 2
    --token-budget)       TOKEN_BUDGET="$2"; shift 2 ;;
    --epochs)             EPOCHS="$2";       shift 2 ;;
    --iters)              ITERS="$2";        shift 2 ;;
    # TIER 3
    --lr)                 LR="$2";           shift 2 ;;
    --min-lr)             MIN_LR="$2";       shift 2 ;;
    --warmup-prct)        WARMUP_PRCT="$2";  shift 2 ;;
    --decay-prct)         DECAY_PRCT="$2";   shift 2 ;;
    # TIER 4
    --global-batch-size)  GBS="$2";          shift 2 ;;
    --micro-batch-size)   MBS="$2";          shift 2 ;;
    --seq-len)            SEQ_LEN="$2";      shift 2 ;;
    # TIER 5
    --simpo-beta)         SIMPO_BETA="$2";        shift 2 ;;
    --simpo-gamma)        SIMPO_GAMMA="$2";       shift 2 ;;
    --simpo-loss-type)    SIMPO_LOSS_TYPE="$2";   shift 2 ;;
    --simpo-sft-weight)   SIMPO_SFT_WEIGHT="$2";  shift 2 ;;
    # TIER 6
    --valid-data-path)    VALID_DATA_PATH="$2";       shift 2 ;;
    --save-path)          SAVE_PATH="$2";             shift 2 ;;
    --save-interval)      SAVE_INTERVAL="$2";         shift 2 ;;
    --eval-interval)      EVAL_INTERVAL="$2";         shift 2 ;;
    --log-interval)       LOG_INTERVAL="$2";          shift 2 ;;
    --eval-iters)         EVAL_ITERS="$2";            shift 2 ;;
    --wandb-project)      WANDB_PROJECT="$2";         shift 2 ;;
    --wandb-exp-name)     WANDB_EXP_NAME="$2";        shift 2 ;;
    --tensorboard-dir)    TENSORBOARD_DIR="$2";       shift 2 ;;
    --tokenizer-type)     TOKENIZER_TYPE="$2";        shift 2 ;;
    --tokenizer-model)    TOKENIZER_MODEL="$2";       shift 2 ;;
    --sft-prompt-format)  SFT_PROMPT_FORMAT="$2";     shift 2 ;;
    --attention-backend)  ATTENTION_BACKEND="$2";     shift 2 ;;
    --recompute-granularity)  RECOMPUTE_GRANULARITY="$2"; shift 2 ;;
    --recompute-method)       RECOMPUTE_METHOD="$2";      shift 2 ;;
    --recompute-num-layers)   RECOMPUTE_NUM_LAYERS="$2";  shift 2 ;;
    --pack-factor)        PACK_FACTOR="$2";           shift 2 ;;
    --split)              SPLIT="$2";                 shift 2 ;;
    --weight-decay)       WEIGHT_DECAY="$2";          shift 2 ;;
    --tp-size)            TP_OVERRIDE="$2";           shift 2 ;;
    --nnodes)             NNODES="$2";                shift 2 ;;
    --node-rank)          NODE_RANK="$2";             shift 2 ;;
    --master-addr)        MASTER_ADDR="$2";           shift 2 ;;
    --master-port)        MASTER_PORT="$2";           shift 2 ;;
    --fused-linear-cross-entropy)    FUSED_LINEAR_CROSS_ENTROPY=true;  shift 1 ;;
    --no-fused-linear-cross-entropy) FUSED_LINEAR_CROSS_ENTROPY=false; shift 1 ;;
    --linear-ce-filter-e-grad)       LINEAR_CE_FILTER_E_GRAD=true;     shift 1 ;;
    --linear-ce-filter-c-grad)       LINEAR_CE_FILTER_C_GRAD=true;     shift 1 ;;
    --no-linear-ce-filter-e-grad)    LINEAR_CE_FILTER_E_GRAD=false;    shift 1 ;;
    --no-linear-ce-filter-c-grad)    LINEAR_CE_FILTER_C_GRAD=false;    shift 1 ;;
    --linear-ce-filter-eps)          LINEAR_CE_FILTER_EPS="$2";        shift 2 ;;
    --log-throughput)     LOG_THROUGHPUT=true; shift 1 ;;
    *) echo "ERROR: Unknown parameter: $1"; exit 1 ;;
  esac
done

# ============================================================================
# TIER 1: Validate required args
# ============================================================================
[[ -z "$MCORE_PATH" ]] && echo "ERROR: --mcore-path is required." && exit 1
[[ -z "$DATA_PATH" ]]  && echo "ERROR: --data-path is required."  && exit 1

if [[ "$MODE" != "cpt" && "$MODE" != "sft" && "$MODE" != "simpo" ]]; then
    echo "ERROR: Unknown --mode '$MODE'. Must be cpt, sft, or simpo."
    exit 1
fi

# ============================================================================
# TIER 6: Tensorboard timestamp
# ============================================================================
if [[ -z "$TENSORBOARD_DIR" ]]; then
    IST_DATE=$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)
    TENSORBOARD_DIR="/home/jovyan/logs/tb/${MODE}_${IST_DATE}"
fi
mkdir -p "$TENSORBOARD_DIR"

# ============================================================================
# TIER 6: Default tokenizer per mode
# ============================================================================
if [[ -z "$TOKENIZER_TYPE" ]]; then
    if [[ "$MODE" == "cpt" ]]; then
        TOKENIZER_TYPE="HuggingFaceTokenizer"
    else
        TOKENIZER_TYPE="SFTTokenizer"
    fi
fi

if [[ -z "$TOKENIZER_MODEL" ]]; then
    TOKENIZER_MODEL="/home/jovyan/models/gemma-3-${MODEL_SIZE}-pt"
    echo "INFO: No --tokenizer-model specified. Using default: $TOKENIZER_MODEL"
fi

# ============================================================================
# TIER 6: Default save path
# ============================================================================
if [[ -z "$SAVE_PATH" ]]; then
    SAVE_PATH="/home/jovyan/data/checkpoints/gemma3-${MODEL_SIZE}-${MODE}"
    echo "INFO: No --save-path specified. Using default: $SAVE_PATH"
fi
mkdir -p "$SAVE_PATH"

# ============================================================================
# TIER 4: Default GBS per mode
# ============================================================================
if [[ $GBS -eq 0 ]]; then
    if [[ "$MODE" == "cpt" ]]; then
        GBS=64
    else
        GBS=32
    fi
fi

# ============================================================================
# TIER 2 + TIER 3: Mode-aware budget, LR, and schedule
# ============================================================================
if [[ "$MODE" == "cpt" ]]; then

    # ── CPT ──────────────────────────────────────────────────────────────────
    # ITERS from token budget (unless --iters hard override is set)
    if [[ $ITERS -eq 0 ]]; then
        ITERS=$(( TOKEN_BUDGET / (SEQ_LEN * GBS) ))
        [[ $(( TOKEN_BUDGET % (SEQ_LEN * GBS) )) -ne 0 ]] && ITERS=$(( ITERS + 1 ))
    fi

    # LR defaults
    [[ "$LR"     == "0" ]] && LR=1e-5
    [[ "$MIN_LR" == "0" ]] && MIN_LR=$(python3 -c "print(f'{float($LR) * 0.1:.2g}')")
    [[ "$WARMUP_PRCT" == "auto" ]] && WARMUP_PRCT=2

    # Validation: no valid-data-path → use split 100,0,0, no eval
    if [[ -z "$VALID_DATA_PATH" ]]; then
        [[ "$SPLIT" == "auto" ]] && SPLIT="100,0,0"
        EVAL_INTERVAL=0
    fi

    [[ "$WANDB_PROJECT" == "AUTO" ]] && WANDB_PROJECT="gemma3-medical-cpt"

elif [[ "$MODE" == "sft" ]]; then

    # ── SFT ──────────────────────────────────────────────────────────────────
    # Packing is always on. ITERS from token scanner (unless --iters override).
    if [[ $ITERS -eq 0 ]]; then
        echo "INFO: Calculating exact training iterations via token scanner..."
        ITERS=$(python3 examples/gemma3/utils/get_sft_tokens.py \
            "$DATA_PATH" "$TOKENIZER_MODEL" "$EPOCHS" "$GBS" "$SEQ_LEN")
        echo "INFO: Token scanner → $ITERS iterations for $EPOCHS epoch(s)."
    fi

    # LR defaults
    [[ "$LR"     == "0" ]] && LR=5e-6
    [[ "$MIN_LR" == "0" ]] && MIN_LR=$(python3 -c "print(f'{float($LR) * 0.1:.2g}')")
    [[ "$WARMUP_PRCT" == "auto" ]] && WARMUP_PRCT=5

    # No valid-data-path → no eval
    [[ -z "$VALID_DATA_PATH" ]] && EVAL_INTERVAL=0

    [[ "$WANDB_PROJECT" == "AUTO" ]] && WANDB_PROJECT="gemma3-medical-sft"

elif [[ "$MODE" == "simpo" ]]; then

    # ── SimPO ─────────────────────────────────────────────────────────────────
    # Packing always on. ITERS from token scanner (unless --iters override).
    if [[ $ITERS -eq 0 ]]; then
        echo "INFO: Calculating exact training iterations via token scanner..."
        ITERS=$(python3 examples/gemma3/utils/get_sft_tokens.py \
            "$DATA_PATH" "$TOKENIZER_MODEL" "$EPOCHS" "$GBS" "$SEQ_LEN")
        echo "INFO: Token scanner → $ITERS iterations for $EPOCHS epoch(s)."
    fi

    # LR defaults (lower than SFT — SimPO paper recommendation)
    [[ "$LR"     == "0" ]] && LR=1e-6
    [[ "$MIN_LR" == "0" ]] && MIN_LR=$(python3 -c "print(f'{float($LR) * 0.1:.2g}')")
    [[ "$WARMUP_PRCT" == "auto" ]] && WARMUP_PRCT=5

    # No valid-data-path → no eval
    [[ -z "$VALID_DATA_PATH" ]] && EVAL_INTERVAL=0

    # SimPO cannot use fused linear cross-entropy (needs full logits for log-probs)
    FUSED_LINEAR_CROSS_ENTROPY=false

    [[ "$WANDB_PROJECT" == "AUTO" ]] && WANDB_PROJECT="gemma3-medical-simpo"

fi

# ============================================================================
# TIER 3: Derive warmup, decay, save, eval from final ITERS
# ============================================================================
WARMUP_ITERS=$(( ITERS * WARMUP_PRCT / 100 ))
[[ $WARMUP_ITERS -eq 0 ]] && WARMUP_ITERS=1

DECAY_ITERS=$(( ITERS * DECAY_PRCT / 100 ))
[[ $DECAY_ITERS -eq 0 ]] && DECAY_ITERS=$ITERS

# Save checkpoint every 20% of total run
[[ $SAVE_INTERVAL -eq 0 ]] && SAVE_INTERVAL=$(( ITERS * 20 / 100 ))
[[ $SAVE_INTERVAL -eq 0 ]] && SAVE_INTERVAL=10

# Eval at half the save interval — only meaningful when validation data exists
if [[ $EVAL_INTERVAL -eq 0 && -n "$VALID_DATA_PATH" ]]; then
    EVAL_INTERVAL=$(( SAVE_INTERVAL / 2 ))
    [[ $EVAL_INTERVAL -eq 0 ]] && EVAL_INTERVAL=5
fi

# ============================================================================
# TIER 6: WandB cleanup
# ============================================================================
if [[ "$WANDB_PROJECT" == "NONE" || -z "$WANDB_PROJECT" ]]; then
    WANDB_PROJECT=""
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
    *) echo "ERROR: Unsupported model size: $MODEL_SIZE. Must be 1b, 4b, or 12b."; exit 1 ;;
esac

if [[ $TP_OVERRIDE -gt 0 ]]; then
    echo "INFO: Overriding default TP=$TP with manual --tp-size=$TP_OVERRIDE"
    TP=$TP_OVERRIDE
fi

# ============================================================================
# GPU Detection & Parallelism Safety
# ============================================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)

if [[ "$NUM_GPUS" -lt "$((TP * PP))" ]]; then
    echo "ERROR: TP=$TP × PP=$PP requires at least $((TP * PP)) GPUs, but only $NUM_GPUS detected."
    echo "Please check CUDA_VISIBLE_DEVICES or adjust --tp-size."
    exit 1
fi

DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPUS --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# ============================================================================
# Recompute Logic
# ============================================================================
if [[ "$RECOMPUTE_GRANULARITY" == "auto" ]]; then
    if [[ $SEQ_LEN -le 8192 ]]; then
        RECOMPUTE_GRANULARITY="none"; RECOMPUTE_METHOD=""; RECOMPUTE_NUM_LAYERS=""
    else
        RECOMPUTE_GRANULARITY="selective"; RECOMPUTE_METHOD=""; RECOMPUTE_NUM_LAYERS=""
    fi
elif [[ "$RECOMPUTE_GRANULARITY" == "full" ]]; then
    [[ "$RECOMPUTE_METHOD"     == "auto" ]] && RECOMPUTE_METHOD="uniform"
    [[ "$RECOMPUTE_NUM_LAYERS" == "auto" ]] && RECOMPUTE_NUM_LAYERS="2"
else
    RECOMPUTE_METHOD=""
    RECOMPUTE_NUM_LAYERS=""
fi

RECOMPUTE_ARGS=""
[[ "$RECOMPUTE_GRANULARITY" != "none" ]] && RECOMPUTE_ARGS="--recompute-granularity $RECOMPUTE_GRANULARITY"
[[ -n "$RECOMPUTE_METHOD"     ]] && RECOMPUTE_ARGS="$RECOMPUTE_ARGS --recompute-method $RECOMPUTE_METHOD"
[[ -n "$RECOMPUTE_NUM_LAYERS" ]] && RECOMPUTE_ARGS="$RECOMPUTE_ARGS --recompute-num-layers $RECOMPUTE_NUM_LAYERS"

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
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --lr-decay-iters $DECAY_ITERS
    --lr-warmup-iters $WARMUP_ITERS
    --weight-decay $WEIGHT_DECAY
    --clip-grad 1.0

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
# Training Args
# ============================================================================
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
    --eval-iters $EVAL_ITERS \
    --no-load-optim \
    --no-load-rng \
    --finetune
"

# ============================================================================
# Data Args
# ============================================================================
if [[ -n "$VALID_DATA_PATH" ]]; then
    # Explicit validation file → works for all modes
    DATA_ARGS="--train-data-path 1.0 $DATA_PATH --valid-data-path 1.0 $VALID_DATA_PATH"
else
    DATA_ARGS="--data-path 1.0 $DATA_PATH"
    # CPT only: pass split (default 100,0,0; expert can override with --split)
    if [[ "$MODE" == "cpt" && "$SPLIT" != "auto" ]]; then
        DATA_ARGS="$DATA_ARGS --split $SPLIT"
    fi
fi

# Mode-specific data flags
if [[ "$MODE" == "sft" ]]; then
    DATA_ARGS="$DATA_ARGS --sft"
    TRAIN_ARGS="$TRAIN_ARGS --eod-mask-loss --no-create-attention-mask-in-dataloader"
elif [[ "$MODE" == "simpo" ]]; then
    DATA_ARGS="$DATA_ARGS --simpo --sft \
        --simpo-beta $SIMPO_BETA \
        --simpo-gamma $SIMPO_GAMMA \
        --simpo-loss-type $SIMPO_LOSS_TYPE \
        --simpo-sft-weight $SIMPO_SFT_WEIGHT"
    TRAIN_ARGS="$TRAIN_ARGS --eod-mask-loss --no-create-attention-mask-in-dataloader"
fi

# ============================================================================
# WandB Args
# ============================================================================
WANDB_ARGS=""
if [[ -n "$WANDB_PROJECT" ]]; then
    WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-exp-name ${WANDB_EXP_NAME:-gemma3-${MODEL_SIZE}-${MODE}}"
fi

# ============================================================================
# Extra Args (CCE loss, throughput, packing)
# ============================================================================
EXTRA_ARGS=""
if [ "$FUSED_LINEAR_CROSS_ENTROPY" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --fused-linear-cross-entropy"
    [ "$LINEAR_CE_FILTER_E_GRAD" = false ] && EXTRA_ARGS="$EXTRA_ARGS --no-linear-ce-filter-e-grad"
    [ "$LINEAR_CE_FILTER_C_GRAD" = false ] && EXTRA_ARGS="$EXTRA_ARGS --no-linear-ce-filter-c-grad"
    [[ "$LINEAR_CE_FILTER_EPS" != "auto" ]] && EXTRA_ARGS="$EXTRA_ARGS --linear-ce-filter-eps $LINEAR_CE_FILTER_EPS"
fi
[ "$LOG_THROUGHPUT" = true ] && EXTRA_ARGS="$EXTRA_ARGS --log-throughput"

# SFT and SimPO always use sequence packing
if [[ "$MODE" == "sft" || "$MODE" == "simpo" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --pack-samples"
    [[ -n "$PACK_FACTOR" ]] && EXTRA_ARGS="$EXTRA_ARGS --pack-factor $PACK_FACTOR"
fi

# ============================================================================
# Chat Template (SFT + SimPO both use SFTTokenizer with tokenize_conversation)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_LAUNCH_ARGS=()
if [[ "$MODE" == "sft" || "$MODE" == "simpo" ]]; then
    TEMPLATE_PATH="$SCRIPT_DIR/utils/gemma3_chat_template.jinja"
    if [[ -f "$TEMPLATE_PATH" ]]; then
        CHAT_TEMPLATE=$(cat "$TEMPLATE_PATH" | tr -d '\n')
        EXTRA_LAUNCH_ARGS+=("--chat-template" "$CHAT_TEMPLATE")
    else
        echo "WARNING: Chat template not found at $TEMPLATE_PATH"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo "================================================================="
echo "  Mode:         $MODE"
echo "  Model:        Gemma 3 $MODEL_SIZE"
echo "  Cluster:      $NNODES Node(s) | Node Rank: $NODE_RANK"
echo "  GPUs/Node:    $NUM_GPUS (TP=$TP, PP=$PP)"
echo "  Seq Length:   $SEQ_LEN"
echo "  GBS:          $GBS (MBS=$MBS, Accum Steps=$((GBS / MBS / NUM_GPUS)))"
echo "  Iters:        $ITERS"
echo "  Recompute:    $RECOMPUTE_GRANULARITY"
echo "  LR:           $LR → $MIN_LR (warmup=$WARMUP_ITERS iters, decay=$DECAY_ITERS iters)"
echo "  Checkpoints:  $SAVE_PATH (every $SAVE_INTERVAL iters)"
echo "  Validation:   ${VALID_DATA_PATH:-DISABLED} (eval every ${EVAL_INTERVAL} iters)"
echo "  CCE Loss:     $FUSED_LINEAR_CROSS_ENTROPY"
echo "  Tokenizer:    $TOKENIZER_TYPE ($SFT_PROMPT_FORMAT)"
echo "  WandB:        ${WANDB_PROJECT:-DISABLED}"
echo "================================================================="

# ============================================================================
# Launch
# ============================================================================
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
