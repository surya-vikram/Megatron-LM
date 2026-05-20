#!/bin/bash
# setup_hopper_container.sh
# Automates the setup for Megatron-LM with FlashAttention-3 on NVIDIA Hopper (H100/H200) containers.

set -e

echo "--- Starting Hopper Setup for Megatron-LM ---"

# 1. Environment Audit
echo "Step 1: Auditing Environment..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Ensure you are on a GPU node."
else
    CAPABILITY=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || echo "0")
    if [ "$CAPABILITY" -eq 9 ]; then
        echo "Check: Hopper GPU detected (Compute Capability 9.x). [OK]"
    else
        echo "WARNING: GPU is not Hopper (Compute Capability $CAPABILITY). FlashAttention-3 may not work."
    fi
fi

CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "Check: CUDA Runtime is $CUDA_VER."

# 2. Install uv
echo "Step 2: Installing uv (Fast Package Manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
# Ensure /root/.local/bin is in path if running as root
if [ "$(id -u)" -eq 0 ]; then
    export PATH=/root/.local/bin:$PATH
fi
# For Jovyan specific paths (backward compatibility)
export PATH=/home/jovyan/.local/bin:$PATH

# 3. Install Megatron-Core
echo "Step 3: Installing Megatron-Core dependencies..."
export UV_BREAK_SYSTEM_PACKAGES=1
uv pip install --system -e .[training,dev]

# 4. Install FlashAttention-3 from Custom Wheel
echo "Step 4: Installing FlashAttention-3 from HuggingFace artifact..."
# Using the custom wheel built on H200 (CUDA 13.1, Python 3.12)
WHEEL_URL="https://huggingface.co/datasets/surya-vikram/hopper-wheels/resolve/main/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl"
pip install "$WHEEL_URL" --no-deps

# 5. Patch TransformerEngine
echo "Step 5: Applying TransformerEngine Backend Patch..."
TE_PATH=$(python3 -c "import transformer_engine; import os; print(os.path.dirname(transformer_engine.__file__))" 2>/dev/null || echo "")

if [ -n "$TE_PATH" ]; then
    BACKEND_FILE="$TE_PATH/pytorch/attention/dot_product_attention/backends.py"
    if [ -f "$BACKEND_FILE" ]; then
        echo "Found TE backend at: $BACKEND_FILE"
        # Patching: from flash_attn_3.flash_attn_interface -> from flash_attn_interface
        sed -i 's/from flash_attn_3.flash_attn_interface/from flash_attn_interface/g' "$BACKEND_FILE"
        echo "Patch Applied. [OK]"
    else
        echo "WARNING: Could not find TE backend file to patch. Manual verification required."
    fi
else
    echo "WARNING: transformer_engine not found. Skipping patch."
fi

# 6. Verification
echo "Step 6: Verifying Setup..."
python3 - <<'EOF'
try:
    import torch
    from transformer_engine.pytorch.attention.dot_product_attention.backends import flash_attn_func_v3
    print(f"Check: FlashAttn V3 Backend loaded: {flash_attn_func_v3 is not None}")
    import os
    os.environ["NVTE_FLASH_ATTN"] = "1"
    print("Check: NVTE_FLASH_ATTN is active.")
    print("--- SETUP COMPLETE: Ready for Hopper Training ---")
except Exception as e:
    print(f"ERROR: Verification failed: {e}")
EOF
