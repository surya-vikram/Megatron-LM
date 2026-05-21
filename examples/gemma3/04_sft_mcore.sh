#!/bin/bash
set -euo pipefail

# SFT Train Gemma3 using Megatron-LM SFTDataset.
# Usage: ./04_sft_mcore.sh <checkpoint_path> <data_path>

CHECKPOINT_PATH="${1:-NO_VALUE_PROVIDED}"
DATA_PATH="${2:-NO_VALUE_PROVIDED}"

if [ "$CHECKPOINT_PATH" = "NO_VALUE_PROVIDED" ] || [ "$DATA_PATH" = "NO_VALUE_PROVIDED" ]; then
    echo "Error: Checkpoint path and Data path are required."
    echo "Usage: $0 <checkpoint_path> <data_path>"
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --master_addr localhost
    --master_port 6000
    --node_rank 0
)

# Gemma3 1B Config
NUM_LAYERS=26
HIDDEN_SIZE=1152
NUM_ATTN_HEADS=4
NUM_QUERY_GROUPS=1
FFN_HIDDEN_SIZE=6912
WINDOW_SIZE=512
VOCAB_SIZE=262144

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS
    --num-query-groups $NUM_QUERY_GROUPS
    --group-query-attention
    --kv-channels 256
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --seq-length 16384
    --max-position-embeddings 16384
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --disable-bias-linear
    --apply-layernorm-1p
    --no-rope-fusion
    --transformer-impl transformer_engine
    --attention-backend flash
    --attention-softmax-in-fp32
    --window-size "$WINDOW_SIZE,$WINDOW_SIZE"
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 20
    --lr 5e-6
    --lr-decay-style constant
    --min-lr 5e-6
    --weight-decay 0.0
    --clip-grad 1.0
    --bf16
    --num-workers 8
)

DATA_ARGS=(
    --sft
    --data-path "$DATA_PATH"
    --tokenizer-type SFTTokenizer
    --sft-tokenizer-prompt-format gemma3
    --tokenizer-model google/gemma-3-1b-it
    --no-create-attention-mask-in-dataloader 
    --vocab-size $VOCAB_SIZE
    --split 100,0,0
)

LOGGING_ARGS=(
    --load "$CHECKPOINT_PATH"
    --save "${CHECKPOINT_PATH}_sft"
    --log-interval 1
    --save-interval 50
    --eval-interval 1000
    --eval-iters 0
    --no-load-optim
    --no-load-rng
)

python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} examples/gemma3/pretrain_gemma3_mcore.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${LOGGING_ARGS[@]}
