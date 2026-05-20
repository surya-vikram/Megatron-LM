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

# Gemma3 4B Config (Example)
MODEL_ARGS=(
    --num-layers 34
    --hidden-size 2560
    --num-attention-heads 8
    --num-query-groups 4
    --group-query-attention
    --kv-channels 256
    --ffn-hidden-size 10240
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
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --train-iters 50
    --lr 1e-4
    --lr-decay-style constant
    --min-lr 1e-4
    --weight-decay 0.0
    --clip-grad 1.0
    --bf16
)

if [ "$DATA_PATH" = "MOCK" ]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 262208
        --split 100,0,0
    )
else
    DATA_ARGS=(
        --data-path "$DATA_PATH"
        --tokenizer-type NullTokenizer
        --vocab-size 262208
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
