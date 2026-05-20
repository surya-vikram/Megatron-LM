#!/bin/bash
set -e
# Install evaluate and jiwer for WER
source /home/jovyan/Megatron-Bridge/.venv/bin/activate
pip install evaluate jiwer

python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import json

# Paths
base_model_path = \"/home/jovyan/models/gemma-3-1b-pt-hf\"
trained_model_path = \"/home/jovyan/models/gemma-3-1b-final-hf/gemma-3-1b-pt-hf\"
with open(\"/home/jovyan/data/medical_corpus/medical_train.jsonl\", \"r\") as f:
    ground_truth = json.load(f)[\"text\"]

# Load Models
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).cuda()
trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path, torch_dtype=torch.bfloat16).cuda()

def get_generation(model, prompt):
    inputs = tokenizer(prompt, return_tensors=\"pt\").to(\"cuda\")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = \"The patient is a 65-year-old male\" # Truncated start of data
wer = evaluate.load(\"wer\")

print(\"Evaluating Base Model...\")
base_gen = get_generation(base_model, prompt)
base_wer = wer.compute(predictions=[base_gen], references=[ground_truth])

print(\"Evaluating Trained Model...\")
trained_gen = get_generation(trained_model, prompt)
trained_wer = wer.compute(predictions=[trained_gen], references=[ground_truth])

print(f\"Base WER: {base_wer}\")
print(f\"Trained WER: {trained_wer}\")
"
