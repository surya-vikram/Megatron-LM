#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NVTE_FLASH_ATTN=1
export PYTHONPATH=$PYTHONPATH:.

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fp8"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters (Restored to 8B - Lion Memory-Saver Mode)
TP_SIZE=1     
CP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
NUM_LAYERS=32
DTYPE="bf16"
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

# Data cache path
DATA_CACHE_PATH="${PWD}/benchmark_cache_llama3_8b_lion"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0134
    --attention-backend flash
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear 
    --recompute-activations
    --recompute-granularity full
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 100000
    --lr-decay-samples 90000
    --lr-warmup-samples 10000
    --lr 0.00015
    --min-lr 0.00001
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
    --exit-duration-in-mins 235 
    --optimizer lion # Use Lion to save 32GB VRAM (removes 1 momentum state)
)

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
)

# Enable sequence parallelism only if TP > 1
if [ $TP_SIZE -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi

# Distributed Data Parallel (DDP) arguments - NO DISTRIBUTED OPTIMIZER for simplest 1-GPU run
DDP_ARGS=(
    # --use-distributed-optimizer 
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments
DATA_ARGS_LIST=(
    "--mock-data"
    "--tokenizer-type NullTokenizer"
    "--vocab-size 128256" 
    "--data-cache-path ${DATA_CACHE_PATH}"
    "--tiktoken-pattern v2" 
    "--split '99,1,0'"
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    "--num-workers 1"
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x
