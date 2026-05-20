#!/bin/bash
# sanity_check.sh
# One-click script to setup, configure, and verify the Megatron-LM Hopper environment.

set -e

echo "=== Megatron-LM Hopper Sanity Check ==="

# 1. Run the setup script
if [ -f "setup_hopper_container.sh" ]; then
    echo "--- Running Environment Setup ---"
    bash setup_hopper_container.sh
else
    echo "ERROR: setup_hopper_container.sh not found!"
    exit 1
fi

# 2. Detect GPU count
TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Sanity check requires at least 1 GPU."
    exit 1
fi
echo "System has $TOTAL_GPUS GPU(s)."

# 3. Configure parallelism and run count
TRAIN_SCRIPT="examples/run_simple_mcore_train_loop.py"
if [ -f "$TRAIN_SCRIPT" ]; then
    echo "--- Configuring $TRAIN_SCRIPT ---"
    
    if [ "$TOTAL_GPUS" -eq 1 ]; then
        echo "Using 1 GPU for sanity check (Single GPU Mode)."
        RUN_COUNT=1
        sed -i 's/tensor_model_parallel_size=2/tensor_model_parallel_size=1/g' "$TRAIN_SCRIPT"
    else
        echo "Multiple GPUs available. Using 2 GPUs for sanity check (Standard Distributed Mode)."
        RUN_COUNT=2
        # Ensure it's set to 2 (default)
        sed -i 's/tensor_model_parallel_size=1/tensor_model_parallel_size=2/g' "$TRAIN_SCRIPT"
    fi
else
    echo "ERROR: $TRAIN_SCRIPT not found!"
    exit 1
fi

# 4. Run the Training Loop
echo "--- Launching Training Loop with FlashAttention-3 (on $RUN_COUNT GPU(s)) ---"
export PYTHONPATH=$PYTHONPATH:.
export NVTE_FLASH_ATTN=1

# Execute on either 1 or 2 GPUs
torchrun --nproc_per_node="$RUN_COUNT" "$TRAIN_SCRIPT"

echo "=== SANITY CHECK COMPLETE ==="
