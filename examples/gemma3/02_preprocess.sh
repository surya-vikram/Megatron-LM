#!/bin/bash
set -euo pipefail

export PYTHONPATH="/home/jovyan/Megatron-Bridge/src:/root/Megatron-LM:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/jovyan/data/corpus}"
JSONL_FILE="${JSONL_FILE:-${DATA_DIR}/train.jsonl}"
RAW_FILE="${RAW_FILE:-${DATA_DIR}/raw_data.txt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DATA_DIR}/corpus_data}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/home/jovyan/models/gemma-3-1b-pt-hf}"
RAW_TEXT_CHARS="${RAW_TEXT_CHARS:-50000}"

mkdir -p "${DATA_DIR}"

if [ ! -f "${JSONL_FILE}" ]; then
    echo "Fetching sample data from internet..."
    curl -L -o "${RAW_FILE}" https://www.gutenberg.org/files/11/11-0.txt
    RAW_FILE="${RAW_FILE}" JSONL_FILE="${JSONL_FILE}" RAW_TEXT_CHARS="${RAW_TEXT_CHARS}" python3 - <<'PY'
import json
import os

raw_file = os.environ["RAW_FILE"]
jsonl_file = os.environ["JSONL_FILE"]
raw_text_chars = int(os.environ["RAW_TEXT_CHARS"])

with open(raw_file, "r") as handle:
    text = handle.read(raw_text_chars)

with open(jsonl_file, "w") as handle:
    handle.write(json.dumps({"text": text.strip()}) + "\n")
PY
fi

python /root/Megatron-LM/tools/preprocess_data.py \
    --input "${JSONL_FILE}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${HF_MODEL_PATH}" \
    --append-eod \
    --json-keys text \
    --workers 1
