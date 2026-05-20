#!/bin/bash
set -e
source /home/jovyan/Megatron-Bridge-Surya/.venv/bin/activate
mkdir -p /home/jovyan/data
python /root/Megatron-LM/tools/preprocess_data.py     --input /home/jovyan/data/medical_data.jsonl     --output-prefix /home/jovyan/data/gemma_medical_data     --tokenizer-type HuggingFaceTokenizer     --tokenizer-model /home/jovyan/models/gemma-3-1b-pt-hf     --append-eod     --workers 1
