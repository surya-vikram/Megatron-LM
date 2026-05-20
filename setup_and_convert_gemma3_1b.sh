#!/bin/bash
set -e

echo "=== Gemma-3 1B Setup and Conversion ==="
export MODEL_ID="google/gemma-3-1b-pt"
export BASE_DIR="/home/jovyan"
export HF_MODEL_PATH="$BASE_DIR/models/gemma-3-1b-pt-hf"
export ROUNDTRIP_PATH="$BASE_DIR/models/gemma-3-1b-roundtrip-test"

echo "1. Syncing Megatron-Bridge to Main branch..."
cd $BASE_DIR/Megatron-Bridge-Surya
git checkout main
git pull origin main

echo "2. Ensuring Python Virtual Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
else
    source .venv/bin/activate
fi

echo "3. Downloading Hugging Face Model ($MODEL_ID)..."
if [ ! -d "$HF_MODEL_PATH" ]; then
    hf download $MODEL_ID --local-dir $HF_MODEL_PATH
fi

echo "4. Running Official Roundtrip Conversion and Verification..."
# Clean up any lingering distributed training ports
pkill -9 -f python || true
pkill -9 -f torchrun || true

export RANK=0 
export WORLD_SIZE=1 
export MASTER_ADDR=localhost 
export MASTER_PORT=12365

# Run the official roundtrip script which creates the megatron model in memory, verifies it, and exports it.
python examples/conversion/hf_megatron_roundtrip.py \
    --hf-model-id $HF_MODEL_PATH \
    --output-dir $ROUNDTRIP_PATH \
    --trust-remote-code

echo "=== Success! 1B Model is verified and ready. ==="
