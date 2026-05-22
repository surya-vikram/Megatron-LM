#!/bin/bash
set -euo pipefail

# Dynamic CPT Script for Gemma 3 (1B, 4B, 12B)
# Optimized for Single-GPU H200 with Precision-Aware Adam

CHECKPOINT_PATH=""
DATA_PATH=""
HF_MODEL_PATH=""
SAVE_PATH=""
TRAIN_ITERS=100
SAVE_INTERVAL=10000
MASTER_PORT=6000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --hf-model-path) HF_MODEL_PATH="$2"; shift 2 ;;
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Dynamic Architecture Inference
read -r NUM_LAYERS HIDDEN_SIZE NUM_ATTN_HEADS NUM_QUERY_GROUPS FFN_HIDDEN_SIZE WINDOW_SIZE VOCAB_SIZE < <(python3 -c "
import json
from pathlib import Path
raw = json.loads(Path('$HF_MODEL_PATH/config.json').read_text())
c = raw.get('text_config', raw)
print(f\"{c.get('num_hidden_layers', 26)} {c.get('hidden_size', 1152)} {c.get('num_attention_heads', 8)} {c.get('num_key_value_heads', 1)} {c.get('intermediate_size', 6912)} {c.get('sliding_window', 512)} {raw.get('vocab_size', 262144)}\")
")

python3 -m torch.distributed.run --nproc_per_node 1 --master_port "$MASTER_PORT" examples/gemma3/pretrain_gemma3_mcore.py \
    --num-layers "$NUM_LAYERS" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-attention-heads "$NUM_ATTN_HEADS" \
    --num-query-groups "$NUM_QUERY_GROUPS" \
    --group-query-attention \
    --kv-channels 256 \
    --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
    --seq-length 2048 \
    --max-position-embeddings 32768 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --qk-layernorm \
    --disable-bias-linear \
    --apply-layernorm-1p \
    --no-rope-fusion \
    --transformer-impl transformer_engine \
    --attention-backend flash \
    --attention-softmax-in-fp32 \
    --window-size "$WINDOW_SIZE,$WINDOW_SIZE" \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters "$TRAIN_ITERS" \
    --lr 5e-6 \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-params-dtype fp16 \
    --exp-avg-dtype fp16 \
    --exp-avg-sq-dtype fp16 \
    --lr-decay-style cosine \
    --bf16 \
    --grad-reduce-in-bf16 \
    --cross-entropy-loss-fusion \
    --empty-unused-memory-level 1 \
    --manual-gc \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --data-path "$DATA_PATH" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$HF_MODEL_PATH" \
    --vocab-size "$VOCAB_SIZE" \
    --split 100,0,0 \
    --load "$CHECKPOINT_PATH" \
    --save "$SAVE_PATH" \
    --tensorboard-dir "${SAVE_PATH}/tensorboard" \
    --log-interval 1 \
    --save-interval "$SAVE_INTERVAL" \
    --eval-interval 1000 \
    --eval-iters 0 \
    --recompute-activations \
    --recompute-granularity full \
    --no-load-optim --no-load-rng
