#!/bin/bash
set -e

# setup_hopper_container.sh: Optimizations for NVIDIA Hopper (H100/H200)

echo "--- Starting Hopper Setup for Megatron-LM ---"

# Find Megatron-LM root relative to this script
MLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$MLM_ROOT"

# 1. Environment Audit
echo "Step 1: Auditing Environment..."
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "Check: CUDA Runtime is $CUDA_VER."

# 2. Install uv
echo "Step 2: Installing uv (Fast Package Manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:/root/.local/bin:/home/jovyan/.local/bin:$PATH

# 3. Install Megatron-Core
echo "Step 3: Installing Megatron-Core dependencies from $MLM_ROOT..."
export UV_BREAK_SYSTEM_PACKAGES=1
uv pip install --system -e .[training,dev]

# 4. Pin FlashInfer
echo "Step 4: Pinning FlashInfer..."
pip install "flashinfer-python==0.6.8.post1" "flashinfer-cubin==0.6.8.post1" --no-deps

# 5. Install FlashAttention-3
echo "Step 5: Installing FlashAttention-3..."
WHEEL_URL="https://huggingface.co/datasets/surya-vikram/hopper-wheels/resolve/main/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl"
pip install "$WHEEL_URL" --no-deps

# 6. Patch TransformerEngine
echo "Step 6: Applying TE Backend Patch..."
TE_PATH=$(python3 -c "import transformer_engine; import os; print(os.path.dirname(transformer_engine.__file__))" 2>/dev/null || echo "")
if [ -n "$TE_PATH" ]; then
    BACKEND_FILE="$TE_PATH/pytorch/attention/dot_product_attention/backends.py"
    if [ -f "$BACKEND_FILE" ]; then
        sed -i 's/from flash_attn_3.flash_attn_interface/from flash_attn_interface/g' "$BACKEND_FILE"
        echo "Patch Applied. [OK]"
    fi
fi

echo "--- Hopper Setup Complete --- "
