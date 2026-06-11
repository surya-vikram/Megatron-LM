#!/bin/bash
set -euo pipefail

# preprocess.sh: Data Preparation for Gemma 3 (CPT and SFT)

usage() {
    echo "Usage: ./preprocess.sh --mode <cpt|sft|simpo> --input <path> [options]"
    echo "Options for CPT:"
    echo "  --output-prefix <path>  Output prefix for .bin and .idx files"
    echo "  --hf-tokenizer <path>   Path to Gemma 3 HF tokenizer"
    echo "  --workers <int>         Number of workers (default: 8)"
    exit 1
}

MODE=""
INPUT=""
OUTPUT_PREFIX=""
HF_TOKENIZER=""
WORKERS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --hf-tokenizer) HF_TOKENIZER="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$MODE" || -z "$INPUT" ]] && usage

if [[ "$MODE" == "cpt" ]]; then
    [[ -z "$OUTPUT_PREFIX" || -z "$HF_TOKENIZER" ]] && usage
    echo "--- Preprocessing CPT Data (Raw Text -> Megatron Binary) --- "
    python3 tools/preprocess_data.py \
        --input "$INPUT" \
        --output-prefix "$OUTPUT_PREFIX" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "$HF_TOKENIZER" \
        --workers "$WORKERS" \
        --append-eod \
        --log-interval 1000
elif [[ "$MODE" == "sft" ]]; then
    echo "--- Validating SFT Instruction Data --- "
    python3 examples/gemma3/utils/validate_sft.py "$INPUT"
elif [[ "$MODE" == "simpo" ]]; then
    echo "--- Validating SimPO Preference Data --- "
    python3 examples/gemma3/utils/validate_simpo.py "$INPUT"
else
    usage
fi
