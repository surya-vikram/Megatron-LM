#!/bin/bash
# remote_test_gemma3_12b.sh
# Comprehensive automation for extraction and parity verification of Gemma 3 12B Text Backbone

set -e

# Configuration
HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
MODEL_ID="google/gemma-3-12b-pt"
REF_MODEL_ID="google/gemma-3-12b-it"
HF_SOURCE_DIR="/home/jovyan/models/gemma-3-12b-pt"
REF_SOURCE_DIR="/home/jovyan/models/gemma-3-12b-it-metadata"
MCORE_TARGET_DIR="/home/jovyan/models/gemma-3-12b-pt-mcore"
STANDALONE_HF_DIR="/home/jovyan/models/gemma-3-12b-standalone-text"

echo "=== Gemma 3 12B Text Extraction Remote Test ==="

# 1. Environment Check
echo "--- Initializing Hopper Environment ---"
cd /root/Megatron-LM

# 2. HF Setup
echo "--- Authenticating with HuggingFace ---"
hf auth login --token $HF_TOKEN

if [ ! -d "$HF_SOURCE_DIR" ]; then
    echo "--- Downloading weights from $MODEL_ID ---"
    mkdir -p "/home/jovyan/models"
    hf download "$MODEL_ID" --local-dir "$HF_SOURCE_DIR"
fi

if [ ! -d "$REF_SOURCE_DIR" ]; then
    echo "--- Downloading metadata only from $REF_MODEL_ID ---"
    # Only download the tiny metadata files to save ~24GB of space
    hf download "$REF_MODEL_ID" --local-dir "$REF_SOURCE_DIR" --include "*.json" "*.model" "*.jinja"
fi

# 3. Extraction
echo "--- Extracting Text Backbone to Megatron-Core ---"
python examples/gemma3/extract_text_backbone.py \
    --hf-model "$HF_SOURCE_DIR" \
    --save-path "$MCORE_TARGET_DIR" \
    --tp-size 1 \
    --pp-size 1

# 4. Standalone Export
echo "--- Exporting MCore Checkpoint back to Standalone HF ---"
python examples/gemma3/export_standalone_text.py \
    --megatron-path "$MCORE_TARGET_DIR" \
    --hf-save-path "$STANDALONE_HF_DIR" \
    --hf-tokenizer-path "$REF_SOURCE_DIR" \
    --tp-size 1 \
    --pp-size 1

# 5. Verification
echo "--- Verifying Logit Parity ---"
python examples/gemma3/verify_standalone_text.py \
    --vlm-path "$HF_SOURCE_DIR" \
    --text-path "$STANDALONE_HF_DIR" \
    --prompt "The capital of France is"

echo ""
echo "=================================================="
echo "SUCCESS: Gemma 3 12B Text extraction verified!"
echo "MCore Path: $MCORE_TARGET_DIR"
echo "Standalone HF Path: $STANDALONE_HF_DIR"
echo "=================================================="
