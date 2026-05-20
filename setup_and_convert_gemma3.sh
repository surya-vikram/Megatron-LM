#!/bin/bash
# setup_and_convert_gemma3.sh
# Automates Megatron-Bridge setup, Gemma-3 4B download, and bidirectional verification.
# Run this after sanity_check.sh.

set -e

# Configuration
BRIDGE_REPO="https://github.com/surya-vikram/Megatron-Bridge.git"
BRIDGE_BRANCH="gemma-patch"
BRIDGE_DIR="$HOME/Megatron-Bridge-Surya"
MODEL_ID="google/gemma-3-4b-pt"
HF_SOURCE_DIR="$HOME/models/gemma-3-4b-pt-hf"
MCORE_TARGET_DIR="$HOME/models/gemma-3-4b-pt-mcore"
HF_ROUNDTRIP_DIR="$HOME/models/gemma-3-4b-pt-roundtrip-hf"

echo "=== Gemma-3 4B Automation Script ==="

# 1. Setup Megatron-Bridge
if [ ! -d "$BRIDGE_DIR" ]; then
    echo "--- Installing Megatron-Bridge ($BRIDGE_BRANCH) ---"
    git clone -b "$BRIDGE_BRANCH" "$BRIDGE_REPO" "$BRIDGE_DIR"
fi

cd "$BRIDGE_DIR"
# Create a symlink to this Megatron-LM repo to ensure the Bridge uses our optimized code
rm -rf 3rdparty/Megatron-LM
# Use absolute path to the current Megatron-LM repo
MEGATRON_LM_DIR=$(cd $(dirname $0)/.. && pwd)
ln -s "$MEGATRON_LM_DIR" 3rdparty/Megatron-LM

# Create safe venv
export UV_HTTP_TIMEOUT=300
if [ ! -d ".venv" ]; then
    uv venv --system-site-packages .venv
fi
source .venv/bin/activate
export UV_BREAK_SYSTEM_PACKAGES=1
uv pip install -e .
cd -

# 2. Download Model
if [ ! -d "$HF_SOURCE_DIR" ]; then
    echo "--- Downloading $MODEL_ID ---"
    mkdir -p "$HOME/models"
    # Assuming user is already logged in via sanity_check or previous steps
    hf download "$MODEL_ID" --local-dir "$HF_SOURCE_DIR"
else
    echo "Check: HF Source model present. [OK]"
fi
# Ensure architectures is correct if previously patched or redownloaded
python3 -c "import json; path = '$HF_SOURCE_DIR/config.json'; c = json.load(open(path)); c['architectures'] = ['Gemma3ForCausalLM']; json.dump(c, open(path, 'w'), indent=2)"

# 3. Conversion & Verification Loop
echo "--- Running Bidirectional Conversion & Parity Check ---"
if [ -d "$BRIDGE_DIR/.venv" ]; then
    source "$BRIDGE_DIR/.venv/bin/activate"
fi

# We must set these for the Bridge's distributed initialization logic
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12356

python "$BRIDGE_DIR/examples/conversion/compare_text_generation.py" \
    --hf-model-id "$HF_SOURCE_DIR" \
    --megatron-path "$MCORE_TARGET_DIR" \
    --hf-save-path "$HF_ROUNDTRIP_DIR" \
    --prompt "The capital of France is" \
    --max-new-tokens 20 \
    --token-compare-method exact \
    --logits-compare-method cosine

echo ""
echo "=================================================="
echo "SUCCESS: Gemma-3 4B is converted and verified!"
echo "Megatron Checkpoint: $MCORE_TARGET_DIR"
echo "=================================================="
