#!/bin/bash
# setup_and_convert_gemma3.sh
# Automates Megatron-Bridge setup, Gemma-3 4B download, and bidirectional verification.
# This script uses specialized stitching logic for Gemma-3 multimodal models.

set -e

# Configuration - Robustly handle HOME
BASE_DIR="${HOME:-/root}"
BRIDGE_REPO="https://github.com/surya-vikram/Megatron-Bridge.git"
BRIDGE_BRANCH="gemma-patch"
BRIDGE_DIR="$BASE_DIR/Megatron-Bridge-Surya"
MODEL_ID="google/gemma-3-4b-pt"
HF_SOURCE_DIR="$BASE_DIR/models/gemma-3-4b-pt-hf"
MCORE_TARGET_DIR="$BASE_DIR/models/gemma-3-4b-pt-mcore"
HF_ROUNDTRIP_DIR="$BASE_DIR/models/gemma-3-4b-pt-roundtrip-hf"

# Find where THIS Megatron-LM repo is located
MEGATRON_LM_DIR=$(cd "$(dirname "$0")" && pwd)

echo "=== Gemma-3 4B Automation Script (Stitching Version) ==="
echo "Base Directory: $BASE_DIR"
echo "Megatron-LM:    $MEGATRON_LM_DIR"

# 1. Setup Megatron-Bridge
if [ ! -d "$BRIDGE_DIR" ]; then
    echo "--- Installing Megatron-Bridge ($BRIDGE_BRANCH) ---"
    git clone -b "$BRIDGE_BRANCH" "$BRIDGE_REPO" "$BRIDGE_DIR"
fi

cd "$BRIDGE_DIR"
# Ensure the symlink is correct and absolute
rm -rf 3rdparty/Megatron-LM
mkdir -p 3rdparty
ln -s "$MEGATRON_LM_DIR" 3rdparty/Megatron-LM
echo "Symlink created: 3rdparty/Megatron-LM -> $MEGATRON_LM_DIR"

# Create safe venv
export UV_HTTP_TIMEOUT=300
if [ ! -d ".venv" ]; then
    uv venv --system-site-packages .venv
fi
source .venv/bin/activate
export UV_BREAK_SYSTEM_PACKAGES=1

# Clean previous build state to avoid metadata name mismatches
rm -rf build/ src/*.egg-info 3rdparty/Megatron-LM/build 3rdparty/Megatron-LM/src/*.egg-info

echo "--- Installing Bridge & Core ---"
uv pip install -e .
cd -

# 2. Download Model
if [ ! -d "$HF_SOURCE_DIR" ]; then
    echo "--- Downloading $MODEL_ID ---"
    mkdir -p "$BASE_DIR/models"
    hf download "$MODEL_ID" --local-dir "$HF_SOURCE_DIR"
else
    echo "Check: HF Source model present. [OK]"
fi

# 3. Conversion Pipeline
echo "--- Step A: Converting HF -> Megatron (Text Backbone Only) ---"
python convert_gemma3.py

echo "--- Step B: Converting Megatron -> HF (Stitching Vision Tower) ---"
python export_gemma3.py

# 4. Verification Loop
echo "--- Step C: Running Parity Check on Stitched Model ---"
if [ -d "$BRIDGE_DIR/.venv" ]; then
    source "$BRIDGE_DIR/.venv/bin/activate"
fi

# Required for distributed init
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12356

# We compare the ORIGINAL HF against the STITCHED HF
# We point --megatron-path to the MCORE dir to satisfy script requirements
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
echo "Stitched HF Model:   $HF_ROUNDTRIP_DIR"
echo "=================================================="
