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

# Default to 1B config if not specified
HF_MODEL_PATH="${3:-google/gemma-3-1b-pt}"

# Dynamic Architecture Inference
if [[ -d "$HF_MODEL_PATH" && -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "Inferring architecture from $HF_MODEL_PATH/config.json ..."
    read -r NUM_LAYERS HIDDEN_SIZE NUM_ATTN_HEADS NUM_QUERY_GROUPS FFN_HIDDEN_SIZE WINDOW_SIZE VOCAB_SIZE < <(python3 -c "
import json
from pathlib import Path
config_raw = json.loads(Path('/config.json').read_text())
config = config_raw.get("text_config", config_raw)
print(f"{config.get('num_hidden_layers', 26)} {config.get('hidden_size', 1152)} {config.get('num_attention_heads', 8)} {config.get('num_key_value_heads', 1)} {config.get('intermediate_size', 6912)} {config.get('sliding_window', 512)} {config_raw.get('vocab_size', 262144)}")
")
else
    echo "Warning: $HF_MODEL_PATH/config.json not found. Falling back to 1B defaults."
    NUM_LAYERS=26
    HIDDEN_SIZE=1152
    NUM_ATTN_HEADS=4
    NUM_QUERY_GROUPS=1
    FFN_HIDDEN_SIZE=6912
    WINDOW_SIZE=512
    VOCAB_SIZE=262144
fi
KV_CHANNELS=256

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
