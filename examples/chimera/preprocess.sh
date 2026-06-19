#!/bin/bash
set -euo pipefail

INPUT="examples/chimera/overfit_doc.jsonl"
OUTPUT_PREFIX="/datasets/megadata/chimera/overfit_doc"
TOKENIZER_MODEL="/datasets/megadata/hf_models/chimera-12b"
WORKERS=8
PYTHON_BIN=""

usage() {
    cat <<'USAGE'
Usage: bash examples/chimera/preprocess.sh [options]

Options:
  --input PATH           Input JSONL with a "text" field.
  --output-prefix PATH   Output prefix before _text_document.{bin,idx}.
  --tokenizer-model PATH HF Chimera tokenizer/model directory.
  --workers N            Number of preprocessing workers.
  --python PATH          Python executable.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --tokenizer-model) TOKENIZER_MODEL="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /workspace/venv/bin/python ]]; then
        PYTHON_BIN="/workspace/venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

[[ -f "$INPUT" ]] || { echo "Missing input JSONL: $INPUT"; exit 1; }
[[ -d "$TOKENIZER_MODEL" || -f "$TOKENIZER_MODEL" ]] || { echo "Missing tokenizer model: $TOKENIZER_MODEL"; exit 1; }

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

"$PYTHON_BIN" tools/preprocess_data.py \
    --input "$INPUT" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --workers "$WORKERS" \
    --append-eod \
    --log-interval 1

echo "Data prefix: ${OUTPUT_PREFIX}_text_document"
