#!/bin/bash
#SBATCH --job-name=gemma3-multinode
#SBATCH --nodes=2                   # Default to 2 nodes, override via CLI if needed
#SBATCH --nodelist=node3,node5      # Target specific cluster nodes (remove or change as needed)
#SBATCH --ntasks-per-node=1         # Run exactly 1 task per node (torchrun handles per-node GPU spawning)
#SBATCH --gres=gpu:8                # 8 GPUs per node (assuming H200/H100/A100 nodes)
#SBATCH --cpus-per-task=32          # Standard CPU allocation per task
#SBATCH --time=24:00:00             # Max run time
#SBATCH --output=gemma3_multinode_%j.out
#SBATCH --error=gemma3_multinode_%j.err

# ==============================================================================
# Production Slurm Multi-Node Batch Submission Script for Gemma 3
# Compatible with CPT / SFT / SimPO modes (1B / 4B / 12B)
# ==============================================================================
#
# How to run:
#   Option A: Edit the default configuration block below and run:
#             sbatch submit_multinode.sbatch
#
#   Option B: Pass arguments dynamically via environment variables:
#             sbatch --export=ALL,MODE=sft,MODEL_SIZE=12b,DATA_PATH=/path/to/data.jsonl \
#                    --nodes=5 --gres=gpu:8 submit_multinode.sbatch
#
# ==============================================================================
# 1. USER CONFIGURATION (Defaults)
# ==============================================================================
MODE="${MODE:-cpt}"                   # cpt | sft | simpo
MODEL_SIZE="${MODEL_SIZE:-4b}"         # 1b | 4b | 12b
MCORE_PATH="${MCORE_PATH:-}"           # Path to MCore checkpoint directory
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"   # Path to HuggingFace tokenizer directory
DATA_PATH="${DATA_PATH:-}"             # CPT bin prefix (no extension) or SFT/SimPO jsonl
VALID_DATA_PATH="${VALID_DATA_PATH:-}"   # Optional validation path (empty to disable)

# Cluster Rendezvous Settings
MASTER_PORT="${MASTER_PORT:-6000}"     # Free network port for master process rendezvous
# ==============================================================================

set -eu
set -o pipefail

# 2. Path Autodetect
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 3. Parameter Validation
if [[ -z "$MCORE_PATH" ]]; then
    echo "ERROR: MCORE_PATH is required but not set." >&2
    exit 1
fi
if [[ -z "$TOKENIZER_MODEL" ]]; then
    echo "ERROR: TOKENIZER_MODEL is required but not set." >&2
    exit 1
fi
if [[ -z "$DATA_PATH" ]]; then
    echo "ERROR: DATA_PATH is required but not set." >&2
    exit 1
fi

# 4. Auto-resolve Cluster Nodes & Master Address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID

# 5. Network Optimization & Debugging Env Variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                # Enable InfiniBand (Set to 1 if using standard Ethernet/RoCE)

# If using Ethernet, automatically bind to correct interface name if specified
# export NCCL_SOCKET_IFNAME=eth0 

echo "====================================================================="
echo "  [SLURM MULTI-NODE RUNTIME ENVIRONMENT]"
echo "  Job ID:           $SLURM_JOB_ID"
echo "  Total Nodes:      $NNODES"
echo "  Rank of Node:     $NODE_RANK"
echo "  Master Node IP:   $MASTER_ADDR"
echo "  Master Port:      $MASTER_PORT"
echo "  Execution Mode:   $MODE"
echo "  Model Size:       $MODEL_SIZE"
echo "====================================================================="

# 6. Execution via srun
# Note: srun distributes this call across all nodes. We pass extra CLI arguments
# using "$@" to allow passing runtime overrides directly to sbatch commands.
srun bash "$SCRIPT_DIR/train.sh" \
    --mode "$MODE" \
    --model-size "$MODEL_SIZE" \
    --mcore-path "$MCORE_PATH" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --data-path "$DATA_PATH" \
    ${VALID_DATA_PATH:+--valid-data-path "$VALID_DATA_PATH"} \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    "$@"
