#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
INPUT="$SCRIPT_DIR/data/pretrain/overfit.jsonl"
OUTPUT_PREFIX="$SCRIPT_DIR/data/pretrain/overfit"
TOKENIZER_MODEL="/datasets/megadata/hf_models/chimera-10b"
WORKERS=8
OPTIMIZED=false
NUM_TOKENIZERS=""
NUM_READERS=""
NUM_WRITERS=""
QUEUE_MEMORY_BUDGET_GB=""
LOG_INTERVAL=1000
LOG_INTERVAL_SECONDS=30
PARQUET_BATCH_SIZE=1024
SKIP_DOCUMENT_COUNT=false
PYTHON_BIN=""

usage() {
    cat <<'USAGE'
Usage: bash examples/chimera/preprocess.sh [options]

Options:
  --input PATH           A .jsonl/.parquet file, glob, or recursively scanned directory.
  --output-prefix PATH   Output prefix before _text_document.{bin,idx}.
  --tokenizer-model PATH HF Chimera tokenizer/model directory.
  --workers N            Legacy preprocessing workers (default: 8).
  --optimized            Use dedicated reader/tokenizer/writer process pools.
  --num-readers N        Reader processes (required with --optimized).
  --num-tokenizers N     Tokenizer processes (required with --optimized).
  --num-writers N        Writer/shard processes (required with --optimized).
  --queue-memory-budget-gb N
                         Queue-sizing memory budget (required with --optimized).
  --log-interval N       Check progress every N documents (default: 1000).
  --log-interval-seconds N
                         Minimum seconds between progress logs (default: 30).
  --parquet-batch-size N Read N parquet rows per batch (default: 1024).
  --skip-document-count  Skip exact totals, percentage, and ETA.
  --python PATH          Python executable.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --tokenizer-model) TOKENIZER_MODEL="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --optimized) OPTIMIZED=true; shift ;;
        --num-tokenizers) NUM_TOKENIZERS="$2"; shift 2 ;;
        --num-readers) NUM_READERS="$2"; shift 2 ;;
        --num-writers) NUM_WRITERS="$2"; shift 2 ;;
        --queue-memory-budget-gb) QUEUE_MEMORY_BUDGET_GB="$2"; shift 2 ;;
        --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
        --log-interval-seconds) LOG_INTERVAL_SECONDS="$2"; shift 2 ;;
        --parquet-batch-size) PARQUET_BATCH_SIZE="$2"; shift 2 ;;
        --skip-document-count) SKIP_DOCUMENT_COUNT=true; shift ;;
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

if [[ ! -e "$INPUT" && "$INPUT" != *'*'* && "$INPUT" != *'?'* && "$INPUT" != *'['* ]]; then
    echo "Missing input path: $INPUT"
    exit 1
fi
[[ -d "$TOKENIZER_MODEL" || -f "$TOKENIZER_MODEL" ]] || { echo "Missing tokenizer model: $TOKENIZER_MODEL"; exit 1; }

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

COUNT_ARGS=()
if [[ "$SKIP_DOCUMENT_COUNT" == true ]]; then
    COUNT_ARGS+=(--skip-document-count)
fi

if [[ "$OPTIMIZED" == true ]]; then
    [[ -n "$NUM_READERS" ]] || { echo "--num-readers is required with --optimized"; exit 1; }
    [[ -n "$NUM_TOKENIZERS" ]] || { echo "--num-tokenizers is required with --optimized"; exit 1; }
    [[ -n "$NUM_WRITERS" ]] || { echo "--num-writers is required with --optimized"; exit 1; }
    [[ -n "$QUEUE_MEMORY_BUDGET_GB" ]] || { echo "--queue-memory-budget-gb is required with --optimized"; exit 1; }
    "$PYTHON_BIN" tools/preprocess_data_pipeline.py \
        --input "$INPUT" \
        --output-prefix "$OUTPUT_PREFIX" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "$TOKENIZER_MODEL" \
        --num-readers "$NUM_READERS" \
        --num-tokenizers "$NUM_TOKENIZERS" \
        --num-writers "$NUM_WRITERS" \
        --queue-memory-budget-gb "$QUEUE_MEMORY_BUDGET_GB" \
        --append-eod \
        --log-interval-seconds "$LOG_INTERVAL_SECONDS"
    echo "Data prefix: ${OUTPUT_PREFIX}_text_document"
    exit 0
fi

"$PYTHON_BIN" tools/preprocess_data.py \
    --input "$INPUT" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --workers "$WORKERS" \
    --parquet-batch-size "$PARQUET_BATCH_SIZE" \
    --append-eod \
    --log-interval "$LOG_INTERVAL" \
    --log-interval-seconds "$LOG_INTERVAL_SECONDS" \
    "${COUNT_ARGS[@]}"

echo "Data prefix: ${OUTPUT_PREFIX}_text_document"
