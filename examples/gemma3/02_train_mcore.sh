#!/bin/bash
set -euo pipefail

# Train Gemma3 using Megatron-LM loop.
# Usage: ./02_train_mcore.sh <checkpoint_path> <data_path>

CHECKPOINT_PATH="${1:-NO_VALUE_PROVIDED}"
DATA_PATH="${2:-MOCK}"

if [ "$CHECKPOINT_PATH" = "NO_VALUE_PROVIDED" ]; then
    echo "Error: Checkpoint path is required."
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --master_addr localhost
    --master_port 6000
    --node_rank 0
)

# Default to 1B config if not specified as second argument
MODEL_SIZE="${2:-1B}"

if [ "$MODEL_SIZE" = "1B" ]; then
    # Gemma3 1B Config
    NUM_LAYERS=26
    HIDDEN_SIZE=1152
    NUM_ATTN_HEADS=4
    NUM_QUERY_GROUPS=1
    FFN_HIDDEN_SIZE=6912
    WINDOW_SIZE=512
    VOCAB_SIZE=262144
    KV_CHANNELS=256
elif [ "$MODEL_SIZE" = "4B" ]; then
    # Gemma3 4B Config
    NUM_LAYERS=34
    HIDDEN_SIZE=3072
    NUM_ATTN_HEADS=12
    NUM_QUERY_GROUPS=4
    FFN_HIDDEN_SIZE=10240
    WINDOW_SIZE=1024
    VOCAB_SIZE=262208
    KV_CHANNELS=256
elif [ "$MODEL_SIZE" = "12B" ]; then
    # Gemma3 12B Config
    NUM_LAYERS=40
    HIDDEN_SIZE=4096
    NUM_ATTN_HEADS=32
    NUM_QUERY_GROUPS=8
    FFN_HIDDEN_SIZE=15360
    WINDOW_SIZE=1024
    VOCAB_SIZE=262208
    KV_CHANNELS=128
else
    echo "Unknown model size: $MODEL_SIZE. Defaulting to 1B."
    NUM_LAYERS=16
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=8
    NUM_QUERY_GROUPS=1
    FFN_HIDDEN_SIZE=8192
    WINDOW_SIZE=512
    VOCAB_SIZE=262144
    KV_CHANNELS=256
fi

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS
    --num-query-groups $NUM_QUERY_GROUPS
    --group-query-attention
    --kv-channels $KV_CHANNELS
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --seq-length 1024
    --max-position-embeddings 4096
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
    --global-batch-size 1
    --train-iters 5
    --lr 1e-7
    --lr-decay-style constant
    --min-lr 1e-7
    --weight-decay 0.0
    --clip-grad 1.0
    --bf16
)

if [ "$DATA_PATH" = "MOCK" ] || [ "$DATA_PATH" = "1B" ] || [ "$DATA_PATH" = "4B" ]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size $VOCAB_SIZE
        --split 100,0,0
    )
else
    DATA_ARGS=(
        --data-path "$DATA_PATH"
        --tokenizer-type NullTokenizer
        --vocab-size $VOCAB_SIZE
        --split 100,0,0
    )
fi

LOGGING_ARGS=(
    --load "$CHECKPOINT_PATH"
    --save "${CHECKPOINT_PATH}_trained"
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
