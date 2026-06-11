#!/bin/bash
set -eu
set -o pipefail

# ============================================================================
# ★  GLOBAL PATH CONFIGURATION
# ============================================================================
MEGADATA_ROOT="/datasets/megadata"

# train.sh: Production Training Engine for Gemma 3 CPT/SFT/SimPO (1B/4B/12B)
# Optimized for single-node training on NVIDIA H200/H100/A100 GPUs.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 1 — IDENTITY & CHECKPOINTING (must pass/configure for every run)
#   --mode          cpt | sft | simpo
#   --model-size    1b  | 4b  | 12b
#   --mcore-path    checkpoint to load from
#   --data-path     training data (Megatron binary prefix for CPT, JSONL for SFT/SimPO)
#   --save-path     directory to save checkpoint outputs
#   --resume-from-checkpoint      flag to resume a crashed/preempted run
#
# TIER 2 — TRAINING BUDGET & EVAL (one arg per mode; --iters overrides everything)
#   CPT:       --token-budget   tokens to train on  (default: 500M)
#   SFT/SimPO: --epochs         epochs to train for (default: 1.0)
#   All:       --iters          hard step count override (skips auto-calculation)
#   All:       --save-interval, --eval-interval, --eval-iters
#
# TIER 3 — LR & OPTIMIZER    (auto-set per mode, overridable)
#   --lr            CPT: 1e-5 | SFT: 5e-6 | SimPO: 1e-6
#   --min-lr        default: lr × 0.1
#   --warmup-prct   CPT: 2% | SFT/SimPO: 5%
#   --decay-prct    all: 90%
#   --lr-decay-style default: cosine
#   --weight-decay  default: 0.1
#   --clip-grad     gradient clipping limit (default: 1.0)
#   --adam-beta1, --adam-beta2
#
# TIER 4 — BATCH & SEQUENCE (auto per mode, overridable)
#   --global-batch-size   CPT: 64 | SFT/SimPO: 32
#   --micro-batch-size    all: 1
#   --seq-len             all: 8192
#
# TIER 5 — ALGORITHM PARAMS (SimPO only)
#   --simpo-beta, --simpo-gamma, --simpo-loss-type, --simpo-sft-weight
#
# TIER 6 — INFRA & CLUSTER   (expert overrides, all have sensible defaults)
#   --valid-data-path     if not passed → no validation at all
#   --wandb-project, --wandb-exp-name, --tensorboard-dir
#   --tp-size, --nnodes, --node-rank, --master-addr, --master-port
#   --recompute-granularity, --recompute-method, --recompute-num-layers
#   --num-workers         CPU dataloader workers count
#   --pack-factor         expert cap on samples per packed step (default: unlimited)
#   --split               CPT expert override (default: 100,0,0)
#   --tokenizer-type, --tokenizer-model, --sft-prompt-format, --attention-backend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "--- Gemma 3 Production Training Engine ---"

# ============================================================================
# CRITICAL: Force in-order kernel launch & dynamic memory allocation
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ════════════════════════════════════════════════════════════════════════════
# ★  USER CONFIGURATION
#    Edit values directly here — or pass any as a CLI flag to override.
#    CLI flags always win over what's written below.
# ════════════════════════════════════════════════════════════════════════════

# ── Identity & Checkpointing ────────────────────────────────────────────────
MODE="cpt"              # cpt | sft | simpo
MODEL_SIZE="4b"         # 1b  | 4b  | 12b
MCORE_PATH=""           # checkpoint to load from (HF-converted MCore format)
DATA_PATH=""            # training data: Megatron binary prefix (CPT), JSONL (SFT/SimPO)
VALID_DATA_PATH=""      # validation data — leave empty to disable validation entirely
SAVE_PATH=""            # output directory for checkpoints (leave empty for auto path)
RESUME_FROM_CHECKPOINT=false  # set to true to resume crashed/preempted run with optim/rng state
SAVE_WEIGHTS_ONLY=false # set to true to only save model weights and omit optimizer/RNG states (makes checkpoints lean)

# ── Training Budget & Evaluation ─────────────────────────────────────────────
TOKEN_BUDGET=500000000  # CPT:       tokens to train on        (default: 500M)
EPOCHS=1.0              # SFT/SimPO: epochs to train for       (default: 1 epoch)
ITERS=0                 # All modes: hard step override — set > 0 to skip auto-calc
SAVE_INTERVAL=0         # checkpoint frequency (steps). 0 = auto-calculate (every 20%)
EVAL_INTERVAL=0         # validation frequency (steps). 0 = auto-calculate (half of save interval)
EVAL_ITERS=2            # number of validation steps/batches to run during evaluation

# ── Learning Rate & Optimizer ────────────────────────────────────────────────
#    Auto defaults: CPT → 1e-5 | SFT → 5e-6 | SimPO → 1e-6
LR=0                    # peak learning rate
MIN_LR=0                # decay floor  (0 = LR × 0.1)
WARMUP_PRCT="auto"      # warmup % of total iters  (auto: CPT→2%, SFT/SimPO→5%)
DECAY_PRCT=90           # cosine decay over this % of total iters
LR_DECAY_STYLE="cosine" # cosine | linear | constant
WEIGHT_DECAY=0.1        # optimizer weight decay coefficient
CLIP_GRAD="1.0"         # gradient clipping limit
ADAM_BETA1="0.9"        # Adam beta1 optimizer parameter
ADAM_BETA2="0.95"       # Adam beta2 optimizer parameter

# ── Batch & Sequence  (0 = auto-set per mode) ────────────────────────────────
#    Auto defaults: CPT GBS→64 | SFT/SimPO GBS→32
GBS=0                   # global batch size  (tokens/step = GBS × SEQ_LEN)
MBS=1                   # micro-batch size   (reduce if OOM)
SEQ_LEN=8192            # sequence / context length

# ── SimPO Algorithm Params ───────────────────────────────────────────────────
SIMPO_BETA="2.0"        # reward scaling factor
SIMPO_GAMMA="0.5"       # target margin between chosen and rejected
SIMPO_LOSS_TYPE="sigmoid"  # loss function: sigmoid | hinge
SIMPO_SFT_WEIGHT="0.0"  # SFT regularization weight (0 = disabled)
USE_CCE_SIMPO=true      # use Apple's memory-efficient Cut Cross-Entropy (CCE) for SimPO (reduces VRAM)

# ── Compute, Parallelism & Performance ──────────────────────────────────────
#    Auto defaults: 1b/4b → TP=1 PP=1 | 12b → TP=2 PP=1
#    DP is always auto = NUM_GPUS / (TP × PP)
TP_OVERRIDE=0           # Tensor Parallel degree  (0 = auto per model size)
NNODES=1                # number of nodes in cluster
NODE_RANK=0             # rank of THIS node  (0 = master)
MASTER_ADDR="localhost" # master node IP  (change for multi-node)
MASTER_PORT=6789        # distributed rendezvous port
RECOMPUTE_GRANULARITY="auto" # auto (none for <=8k, selective for >8k) | none | selective | full
NUM_WORKERS=4           # number of dataloader CPU workers per GPU

# ── Telemetry & Experiment Tracking (WandB) ──────────────────────────────────
WANDB_PROJECT="NONE"    # Weights & Biases project name ("NONE" = disabled; "AUTO" = auto per mode)
WANDB_EXP_NAME=""       # Weights & Biases experiment run name (leave empty for auto)

# ── Dataset Visibility / Debug ───────────────────────────────────────────────
DEBUG_DATASET=false     # per-step packing trace → stdout rank-0 (very verbose)
LOG_DATASET_STATS=false # aggregate packing stats every 100 steps → stdout rank-0
WARN_OVERSIZED=false    # one-time warning when samples are skipped or malformed

# ════════════════════════════════════════════════════════════════════════════
# ✦  INTERNAL DEFAULTS  —  rarely need changing
# ════════════════════════════════════════════════════════════════════════════
LOG_INTERVAL=1
TENSORBOARD_DIR=""
TOKENIZER_TYPE=""
TOKENIZER_MODEL=""      # HF path for tokenizer fallback if needed (will default based on model size)
SFT_PROMPT_FORMAT="gemma3"
ATTENTION_BACKEND="flash"
RECOMPUTE_METHOD="auto"
RECOMPUTE_NUM_LAYERS="auto"
FUSED_LINEAR_CROSS_ENTROPY=true
LINEAR_CE_FILTER_E_GRAD=false
LINEAR_CE_FILTER_C_GRAD=false
LINEAR_CE_FILTER_EPS="0.0"
LOG_THROUGHPUT=true
PACK_FACTOR=""          # expert: cap samples per packed step (default: unlimited)
SPLIT="auto"            # CPT expert override (default: 100,0,0 when no valid-data-path)


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
    --save-path)          SAVE_PATH="$2";    shift 2 ;;
    --resume-from-checkpoint) RESUME_FROM_CHECKPOINT=true; shift 1 ;;
    --no-resume-from-checkpoint) RESUME_FROM_CHECKPOINT=false; shift 1 ;;
    --save-weights-only)   SAVE_WEIGHTS_ONLY=true;  shift 1 ;;
    --no-save-weights-only) SAVE_WEIGHTS_ONLY=false; shift 1 ;;
    # TIER 2
    --token-budget)       TOKEN_BUDGET="$2"; shift 2 ;;
    --epochs)             EPOCHS="$2";       shift 2 ;;
    --iters)              ITERS="$2";        shift 2 ;;
    --save-interval)      SAVE_INTERVAL="$2";shift 2 ;;
    --eval-interval)      EVAL_INTERVAL="$2";shift 2 ;;
    --eval-iters)         EVAL_ITERS="$2";   shift 2 ;;
    # TIER 3
    --lr)                 LR="$2";           shift 2 ;;
    --min-lr)             MIN_LR="$2";       shift 2 ;;
    --warmup-prct)        WARMUP_PRCT="$2";  shift 2 ;;
    --decay-prct)         DECAY_PRCT="$2";   shift 2 ;;
    --lr-decay-style)     LR_DECAY_STYLE="$2";shift 2 ;;
    --weight-decay)       WEIGHT_DECAY="$2"; shift 2 ;;
    --clip-grad)          CLIP_GRAD="$2";    shift 2 ;;
    --adam-beta1)         ADAM_BETA1="$2";   shift 2 ;;
    --adam-beta2)         ADAM_BETA2="$2";   shift 2 ;;
    # TIER 4
    --global-batch-size)  GBS="$2";          shift 2 ;;
    --micro-batch-size)   MBS="$2";          shift 2 ;;
    --seq-len)            SEQ_LEN="$2";      shift 2 ;;
    # TIER 5
    --simpo-beta)         SIMPO_BETA="$2";        shift 2 ;;
    --simpo-gamma)        SIMPO_GAMMA="$2";       shift 2 ;;
    --simpo-loss-type)    SIMPO_LOSS_TYPE="$2";   shift 2 ;;
    --simpo-sft-weight)   SIMPO_SFT_WEIGHT="$2";  shift 2 ;;
    --use-cce-simpo)      USE_CCE_SIMPO=true;     shift 1 ;;
    --no-use-cce-simpo)   USE_CCE_SIMPO=false;    shift 1 ;;
    # TIER 6
    --valid-data-path)    VALID_DATA_PATH="$2";       shift 2 ;;
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
    --tp-size)            TP_OVERRIDE="$2";           shift 2 ;;
    --nnodes)             NNODES="$2";                shift 2 ;;
    --node-rank)          NODE_RANK="$2";             shift 2 ;;
    --master-addr)        MASTER_ADDR="$2";           shift 2 ;;
    --master-port)        MASTER_PORT="$2";           shift 2 ;;
    --num-workers)        NUM_WORKERS="$2";           shift 2 ;;
    --fused-linear-cross-entropy)    FUSED_LINEAR_CROSS_ENTROPY=true;  shift 1 ;;
    --no-fused-linear-cross-entropy) FUSED_LINEAR_CROSS_ENTROPY=false; shift 1 ;;
    --linear-ce-filter-e-grad)       LINEAR_CE_FILTER_E_GRAD=true;     shift 1 ;;
    --linear-ce-filter-c-grad)       LINEAR_CE_FILTER_C_GRAD=true;     shift 1 ;;
    --no-linear-ce-filter-e-grad)    LINEAR_CE_FILTER_E_GRAD=false;    shift 1 ;;
    --no-linear-ce-filter-c-grad)    LINEAR_CE_FILTER_C_GRAD=false;    shift 1 ;;
    --linear-ce-filter-eps)          LINEAR_CE_FILTER_EPS="$2";        shift 2 ;;
    --log-throughput)     LOG_THROUGHPUT=true; shift 1 ;;
    # Debug / Visibility
    --debug-dataset)          DEBUG_DATASET=true;    shift 1 ;;
    --log-dataset-stats)      LOG_DATASET_STATS=true; shift 1 ;;
    --warn-oversized-samples) WARN_OVERSIZED=true;   shift 1 ;;
    *) echo "ERROR: Unknown parameter: $1"; exit 1 ;;
  esac
done

# ============================================================================
# TIER 1: Validate required args
# ============================================================================
[[ -z "$MCORE_PATH" ]] && echo "ERROR: --mcore-path is required." && exit 1
[[ -z "$TOKENIZER_MODEL" ]] && echo "ERROR: --tokenizer-model is required." && exit 1
[[ -z "$DATA_PATH" ]]  && echo "ERROR: --data-path is required."  && exit 1

if [[ "$MODE" != "cpt" && "$MODE" != "sft" && "$MODE" != "simpo" ]]; then
    echo "ERROR: Unknown --mode '$MODE'. Must be cpt, sft, or simpo."
    exit 1
fi

# ============================================================================
# 2. Execution Workspace (Timestamped)
# ============================================================================
IST_DATE=$(TZ='Asia/Kolkata' date +%Y%m%d_%H%M%S)
RUN_DIR="$MEGADATA_ROOT/training_runs/$IST_DATE"

[[ -z "$SAVE_PATH" ]] && SAVE_PATH="$RUN_DIR/checkpoints"
[[ -z "$TENSORBOARD_DIR" ]] && TENSORBOARD_DIR="$RUN_DIR/logs/tb"

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR"

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
        ITERS=$(python3 examples/gemma3/utils/get_sft_tokens.py             "$DATA_PATH" "$TOKENIZER_MODEL" "$EPOCHS" "$GBS" "$SEQ_LEN")
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
        ITERS=$(python3 examples/gemma3/utils/get_sft_tokens.py             "$DATA_PATH" "$TOKENIZER_MODEL" "$EPOCHS" "$GBS" "$SEQ_LEN")
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
[[ $EVAL_INTERVAL -eq 0 ]] && EVAL_INTERVAL=999999

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

if [[ $TP -gt 1 ]]; then
    echo "INFO: TP=$TP > 1, dynamically enabling sequence parallelism and TP communication overlap."
    MODEL_ARGS="$MODEL_ARGS --sequence-parallel --tp-comm-overlap"
fi

# ============================================================================
# Optimizer Args
# ============================================================================
OPTIM_ARGS="
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style $LR_DECAY_STYLE \
    --lr-decay-iters $DECAY_ITERS \
    --lr-warmup-iters $WARMUP_ITERS \
    --weight-decay $WEIGHT_DECAY \
    --clip-grad $CLIP_GRAD \
    --adam-beta1 $ADAM_BETA1 \
    --adam-beta2 $ADAM_BETA2 \
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
    --num-workers $NUM_WORKERS \
    --manual-gc \
    --manual-gc-interval 5 \
    --overlap-grad-reduce \
    --overlap-param-gather
"

# ============================================================================
# Logging & Checkpointing Args (Dynamic Resume support)
# ============================================================================
LOG_ARGS="
    --load $MCORE_PATH \
    --save $SAVE_PATH \
    --tensorboard-dir $TENSORBOARD_DIR \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --eval-iters $EVAL_ITERS
"
if [ "$RESUME_FROM_CHECKPOINT" = false ]; then
    LOG_ARGS="$LOG_ARGS --no-load-optim --no-load-rng --finetune"
fi
if [ "$SAVE_WEIGHTS_ONLY" = true ]; then
    LOG_ARGS="$LOG_ARGS --no-save-optim --no-save-rng"
fi

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

# SFT and SimPO always use sequence packing unless disabled
if [[ "${DISABLE_PACKING:-}" != "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --pack-samples"
fi
[[ -n "$PACK_FACTOR" ]] && EXTRA_ARGS="$EXTRA_ARGS --pack-factor $PACK_FACTOR"
[ "$USE_CCE_SIMPO" = true ] && EXTRA_ARGS="$EXTRA_ARGS --use-cce-simpo"

# Dataset visibility flags (passed through to dataset classes via args namespace)
[ "$DEBUG_DATASET"    = true ] && EXTRA_ARGS="$EXTRA_ARGS --debug-dataset"
[ "$LOG_DATASET_STATS" = true ] && EXTRA_ARGS="$EXTRA_ARGS --log-dataset-stats"
[ "$WARN_OVERSIZED"   = true ] && EXTRA_ARGS="$EXTRA_ARGS --warn-oversized-samples"

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
python3 -m torch.distributed.run $DISTRIBUTED_ARGS \
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
