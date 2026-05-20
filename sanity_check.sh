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
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Sanity check requires at least 1 GPU."
    exit 1
fi
echo "Detected $GPU_COUNT GPU(s)."

# 3. Configure parallelism in the example script
TRAIN_SCRIPT="examples/run_simple_mcore_train_loop.py"
if [ -f "$TRAIN_SCRIPT" ]; then
    echo "--- Configuring $TRAIN_SCRIPT ---"
    # If 1 GPU, set TP=1.
    if [ "$GPU_COUNT" -eq 1 ]; then
        echo "Setting tensor_model_parallel_size to 1 (Single GPU Mode)."
        sed -i 's/tensor_model_parallel_size=2/tensor_model_parallel_size=1/g' "$TRAIN_SCRIPT"
    else
        echo "Multiple GPUs detected ($GPU_COUNT). Ensuring parallelism fits."
    fi
else
    echo "ERROR: $TRAIN_SCRIPT not found!"
    exit 1
fi

# 4. Run the Training Loop
echo "--- Launching Training Loop with FlashAttention-3 ---"
export PYTHONPATH=$PYTHONPATH:.
export NVTE_FLASH_ATTN=1

# Use torchrun based on detected GPU count
torchrun --nproc_per_node="$GPU_COUNT" "$TRAIN_SCRIPT"

echo "=== SANITY CHECK COMPLETE ==="
