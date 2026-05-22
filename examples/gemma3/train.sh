#!/bin/bash
set -euo pipefail

# train.sh: Modular Training Engine for Gemma 3 (1B/4B/12B)
# Supports Continual Pre-training (CPT) and Instruction Tuning (SFT).

echo "--- Gemma 3 Training Engine Launching ---"

# --- Argument Parsing ---
MODE="cpt"            # cpt or sft
MODEL_SIZE="4b"       # 1b, 4b, 12b
MCORE_PATH=""         # Path to converted MCore weights
DATA_PATH=""          # Path to training .bin prefix
VALID_DATA_PATH=""    # Path to validation .bin prefix (Optional)
SAVE_PATH=""          # Where to save checkpoints
ITERS=5000            # Total iterations
LR=1e-6               # Learning rate
WANDB_PROJECT=""      # Optional WandB logging
WANDB_EXP_NAME=""     # Optional WandB experiment name

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
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-exp-name) WANDB_EXP_NAME="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# --- Architecture Configuration ---
case $MODEL_SIZE in
    "1b")
        LAYERS=26; HIDDEN=1152; HEADS=8; GQA=1; WINDOW=512; TP=1; PP=1
        ;;
    "4b")
        LAYERS=34; HIDDEN=2560; HEADS=8; GQA=4; WINDOW=1024; TP=1; PP=1
        ;;
    "12b")
        LAYERS=48; HIDDEN=3840; HEADS=16; GQA=8; WINDOW=1024; TP=2; PP=1
        ;;
    *) echo "Unsupported model size: $MODEL_SIZE"; exit 1 ;;
esac

# --- Training Logic ---
NUM_GPUS=$(nvidia-smi -L | wc -l)
DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPUS --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"

COMMON_ARGS="
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --num-layers $LAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --group-query-attention \
    --num-query-groups $GQA \
    --sliding-window-size $WINDOW \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --train-iters $ITERS \
    --lr $LR \
    --lr-decay-style cosine \
    --min-lr 1e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --no-gradient-accumulation-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --load $MCORE_PATH \
    --save $SAVE_PATH \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10
"

# Handle Data Path (With or without Validation)
if [[ -n "$VALID_DATA_PATH" ]]; then
    DATA_ARGS="--data-path 1.0 $DATA_PATH --valid-data-path $VALID_DATA_PATH"
else
    DATA_ARGS="--data-path 1.0 $DATA_PATH"
fi

# Handle Mode
if [[ "$MODE" == "sft" ]]; then
    DATA_ARGS="$DATA_ARGS --is-instruction-dataset"
fi

# Handle WandB
if [[ -n "$WANDB_PROJECT" ]]; then
    COMMON_ARGS="$COMMON_ARGS --wandb-project $WANDB_PROJECT --wandb-exp-name ${WANDB_EXP_NAME:-gemma3-$MODEL_SIZE-$MODE}"
fi

# Launch
echo "Starting $MODE training for Gemma 3 $MODEL_SIZE..."
torchrun $DISTRIBUTED_ARGS \
    pretrain_gemma3_mcore.py \
    $COMMON_ARGS \
    $DATA_ARGS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP
