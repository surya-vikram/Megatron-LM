#!/bin/bash
# overfit_gemma3_4b.sh
# Final verified CLI-only smoke test for Gemma-3 4B.

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1

CHECKPOINT_PATH="$HOME/models/gemma-3-4b-pt-mcore"
DATA_PATH="$HOME/data/medical_overfit_text_document"
SAVE_PATH="$HOME/models/gemma-3-4b-overfit-mcore"

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --master_addr localhost
    --master_port 6000
    --node_rank 0
)

# Core Gemma-3 4B Native CLI Mapping
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
    --optimizer lion
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type NullTokenizer
    --vocab-size 262208
    --split 100,0,0
)

LOGGING_ARGS=(
    --load $CHECKPOINT_PATH
    --save $SAVE_PATH
    --log-interval 1
    --save-interval 50
    --eval-interval 1000
    --eval-iters 0
    --no-load-optim
    --no-load-rng
)

python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py     ${MODEL_ARGS[@]}     ${TRAINING_ARGS[@]}     ${DATA_ARGS[@]}     ${LOGGING_ARGS[@]}
