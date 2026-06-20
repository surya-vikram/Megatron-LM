#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MCORE_PATH="/datasets/megadata/chimera_bridge_validation/megatron_import"
DATA_PATH="/datasets/megadata/chimera/overfit_doc_text_document"
TOKENIZER_MODEL="/datasets/megadata/hf_models/chimera-12b"
SAVE_PATH="/datasets/megadata/chimera_runs/overfit/checkpoints"
TENSORBOARD_DIR="/datasets/megadata/chimera_runs/overfit/tensorboard"
DATA_CACHE_PATH="/datasets/megadata/chimera_runs/overfit/data_cache"
PYTHON_BIN=""

TRAIN_ITERS=100
SAVE_INTERVAL=50
EVAL_INTERVAL=1000000
EVAL_ITERS=0
SEQ_LENGTH=512
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=0
LR="2e-4"
MIN_LR="2e-5"
LR_WARMUP_ITERS=0
WEIGHT_DECAY="0.0"
CLIP_GRAD="1.0"

TP_SIZE=1
PP_SIZE=1
EP_SIZE=1
ETP_SIZE=1
CP_SIZE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29591
GPUS_PER_NODE=""
CPU_OFFLOAD=false

usage() {
    cat <<'USAGE'
Usage: bash examples/chimera/train.sh [options]

Options:
  --mcore-path PATH          Megatron-Core checkpoint root to load.
  --data-path PREFIX         Megatron .bin/.idx data prefix without suffix.
  --tokenizer-model PATH     HF Chimera tokenizer/model directory.
  --save-path PATH           Output checkpoint directory.
  --tensorboard-dir PATH     TensorBoard log directory.
  --data-cache-path PATH     Megatron dataset cache directory.
  --train-iters N            Training iterations.
  --save-interval N          Checkpoint save interval.
  --seq-length N             Training sequence length.
  --micro-batch-size N       Micro batch size.
  --global-batch-size N      Global batch size.
  --lr VALUE                 Peak learning rate.
  --min-lr VALUE             Minimum learning rate.
  --tp-size N                Tensor parallel size.
  --pp-size N                Pipeline parallel size.
  --ep-size N                Expert parallel size.
  --expert-tp-size N         Expert tensor parallel size.
  --cp-size N                Context parallel size.
  --gpus-per-node N          Number of visible GPUs for torchrun.
  --cpu-offload              Offload optimizer state to CPU for single-GPU fallback.
  --nnodes N                 Number of nodes.
  --node-rank N              Node rank.
  --master-addr HOST         Distributed master address.
  --master-port PORT         Distributed master port.
  --python PATH              Python executable.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mcore-path) MCORE_PATH="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --tokenizer-model) TOKENIZER_MODEL="$2"; shift 2 ;;
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        --tensorboard-dir) TENSORBOARD_DIR="$2"; shift 2 ;;
        --data-cache-path) DATA_CACHE_PATH="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --seq-length) SEQ_LENGTH="$2"; shift 2 ;;
        --micro-batch-size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --min-lr) MIN_LR="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --pp-size) PP_SIZE="$2"; shift 2 ;;
        --ep-size) EP_SIZE="$2"; shift 2 ;;
        --expert-tp-size) ETP_SIZE="$2"; shift 2 ;;
        --cp-size) CP_SIZE="$2"; shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --cpu-offload) CPU_OFFLOAD=true; shift 1 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /workspace/venv/bin/python ]]; then
        PYTHON_BIN="/workspace/venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi
if [[ "$GLOBAL_BATCH_SIZE" -eq 0 ]]; then
    GLOBAL_BATCH_SIZE=$((GPUS_PER_NODE * NNODES))
fi

[[ -d "$MCORE_PATH" ]] || { echo "Missing MCore checkpoint: $MCORE_PATH"; exit 1; }
[[ -f "${DATA_PATH}.bin" ]] || { echo "Missing data bin: ${DATA_PATH}.bin"; exit 1; }
[[ -f "${DATA_PATH}.idx" ]] || { echo "Missing data idx: ${DATA_PATH}.idx"; exit 1; }
[[ -d "$TOKENIZER_MODEL" || -f "$TOKENIZER_MODEL" ]] || { echo "Missing tokenizer model: $TOKENIZER_MODEL"; exit 1; }

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR" "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 28
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings 32768
    --position-embedding-type yarn
    --rotary-base 10000000
    --rotary-percent 1.0
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --make-vocab-size-divisible-by 128
    --vocab-size 50176
    --bf16
    --use-flash-attn
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-masked-softmax-fusion
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
)

MOE_ARGS=(
    --num-experts 96
    --moe-layer-freq "[0]+[1]*26+[0]"
    --moe-router-topk 8
    --moe-ffn-hidden-size 704
    --moe-shared-expert-intermediate-size 704
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-topk-scaling-factor 1.0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-shared-expert-overlap
    --moe-per-layer-logging
)

DATA_ARGS=(
    --data-path 1.0 "$DATA_PATH"
    --split 100,0,0
    --data-cache-path "$DATA_CACHE_PATH"
    --no-create-attention-mask-in-dataloader
    --num-workers 1
)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style cosine
    --lr-decay-iters "$TRAIN_ITERS"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay "$WEIGHT_DECAY"
    --clip-grad "$CLIP_GRAD"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --attention-softmax-in-fp32
    --manual-gc
    --manual-gc-interval 5
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --main-params-dtype fp16
    --main-grads-dtype bf16
    --grad-reduce-in-bf16
    --exp-avg-dtype fp16
    --exp-avg-sq-dtype fp16
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --expert-model-parallel-size "$EP_SIZE"
    --context-parallel-size "$CP_SIZE"
)
if [[ "$CPU_OFFLOAD" == true ]]; then
    TRAINING_ARGS+=(
        --optimizer-cpu-offload
        --optimizer-offload-fraction 1.0
        --use-torch-optimizer-for-cpu-offload
    )
fi

LOGGING_ARGS=(
    --load "$MCORE_PATH"
    --save "$SAVE_PATH"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --log-interval 1
    --no-load-optim
    --no-load-rng
    --finetune
    --no-save-optim
    --no-save-rng
    --log-throughput
)

echo "Running Chimera overfit training"
echo "  MCore checkpoint: $MCORE_PATH"
echo "  Data prefix:      $DATA_PATH"
echo "  Save path:        $SAVE_PATH"
echo "  GPUs per node:    $GPUS_PER_NODE"
echo "  Parallelism:      TP=$TP_SIZE PP=$PP_SIZE EP=$EP_SIZE ETP=$ETP_SIZE CP=$CP_SIZE"
echo "  CPU offload:      $CPU_OFFLOAD"
echo "  Seq length:       $SEQ_LENGTH"
echo "  Train iters:      $TRAIN_ITERS"

"$PYTHON_BIN" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    examples/chimera/pretrain_chimera.py \
    --chimera-expert-tp-size "$ETP_SIZE" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"
