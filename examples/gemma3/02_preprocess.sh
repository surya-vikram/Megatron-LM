#!/bin/bash
set -e
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

# Data Directory Setup
DATA_DIR="/home/jovyan/data/corpus"
JSONL_FILE="$DATA_DIR/train.jsonl"
RAW_FILE="$DATA_DIR/raw_data.txt"

mkdir -p $DATA_DIR

# 1. Fetch data if not already exists
if [ ! -f "$JSONL_FILE" ]; then
    echo "Fetching sample data from internet..."
    curl -L -o $RAW_FILE https://www.gutenberg.org/files/11/11-0.txt
    python3 -c "
import json
with open(\"$RAW_FILE\", \"r\") as f:
    text = f.read(50000)
data = {\"text\": text.strip()}
with open(\"$JSONL_FILE\", \"w\") as f:
    f.write(json.dumps(data) + \"\n\")
"
fi

# 2. Preprocess into Megatron Indexed Binary format
python /root/Megatron-LM/tools/preprocess_data.py \
    --input $JSONL_FILE \
    --output-prefix "$DATA_DIR/corpus_data" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /home/jovyan/models/gemma-3-1b-pt-hf \
    --append-eod \
    --json-keys text \
    --workers 1
