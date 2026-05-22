#!/bin/bash
set -euo pipefail

# import.sh: VLM to Megatron-Core Text Backbone Converter

usage() {
    echo "Usage: ./import.sh --hf-model <path> --mcore-path <path>"
    exit 1
}

HF_MODEL=""
MCORE_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-model) HF_MODEL="$2"; shift 2 ;;
        --mcore-path) MCORE_PATH="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$HF_MODEL" || -z "$MCORE_PATH" ]] && usage

echo "--- Converting HF Gemma 3 VLM to Megatron-Core --- "
python3 examples/gemma3/utils/extract_text_backbone.py \
    --hf-model "$HF_MODEL" \
    --save-path "$MCORE_PATH" \
    --tp-size 1 --pp-size 1
