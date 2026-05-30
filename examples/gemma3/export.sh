#!/bin/bash
set -euo pipefail

# export.sh: Megatron-Core to HuggingFace Converter (Supports TP > 1)

usage() {
    echo "Usage: ./export.sh --target <text|vlm> --mcore-path <path> --hf-reference <path> --save-path <path> [--tp-size <int>]"
    exit 1
}

TARGET=""
MCORE_PATH=""
HF_REF=""
SAVE_PATH=""
TP_SIZE="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --mcore-path) MCORE_PATH="$2"; shift 2 ;;
        --hf-reference) HF_REF="$2"; shift 2 ;;
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$TARGET" || -z "$MCORE_PATH" || -z "$HF_REF" || -z "$SAVE_PATH" ]] && usage

if [[ "$TARGET" == "text" ]]; then
    echo "--- Exporting to Standalone HF Causal LM (TP=$TP_SIZE) --- "
    python3 -m torch.distributed.run --nproc_per_node "$TP_SIZE" --master_port 29519 \
        examples/gemma3/utils/export_standalone_text.py \
        --megatron-path "$MCORE_PATH" \
        --hf-save-path "$SAVE_PATH" \
        --hf-tokenizer-path "$HF_REF" \
        --tp-size "$TP_SIZE"
elif [[ "$TARGET" == "vlm" ]]; then
    echo "--- Exporting to Stitched Multimodal VLM (TP=$TP_SIZE) --- "
    python3 -m torch.distributed.run --nproc_per_node "$TP_SIZE" --master_port 29519 \
        examples/gemma3/utils/export_stitched_multimodal.py \
        --megatron-path "$MCORE_PATH" \
        --vlm-hf-path "$HF_REF" \
        --output-path "$SAVE_PATH" \
        --tp-size "$TP_SIZE"
else
    usage
fi
