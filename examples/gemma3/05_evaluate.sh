#!/bin/bash
set -euo pipefail

export PYTHONPATH="/home/jovyan/Megatron-Bridge/src:/root/Megatron-LM:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/jovyan/data/corpus}"
JSONL_FILE="${JSONL_FILE:-${DATA_DIR}/train.jsonl}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/home/jovyan/models/gemma-3-1b-pt-hf}"
FINAL_HF_PATH="${FINAL_HF_PATH:-/home/jovyan/models/gemma-3-1b-final-hf}"
PROMPT_CHARS="${PROMPT_CHARS:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

JSONL_FILE="${JSONL_FILE}" HF_MODEL_PATH="${HF_MODEL_PATH}" FINAL_HF_PATH="${FINAL_HF_PATH}" PROMPT_CHARS="${PROMPT_CHARS}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" python3 - <<'PY'
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

jsonl_file = os.environ["JSONL_FILE"]
hf_base = os.environ["HF_MODEL_PATH"]
hf_trained = os.environ["FINAL_HF_PATH"]
prompt_chars = int(os.environ["PROMPT_CHARS"])
max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])

with open(jsonl_file, "r") as handle:
    full_text = json.loads(handle.readline())["text"]

prompt = full_text[:prompt_chars]
tokenizer = AutoTokenizer.from_pretrained(hf_base, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(hf_base, torch_dtype=torch.bfloat16).cuda()
trained_model = AutoModelForCausalLM.from_pretrained(hf_trained, torch_dtype=torch.bfloat16).cuda()


def generate_text(model):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


print(f"Base Output: {generate_text(base_model)}")
print(f"Trained Output: {generate_text(trained_model)}")
PY
