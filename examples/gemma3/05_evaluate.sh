#!/bin/bash
set -e
source /home/jovyan/Megatron-Bridge-Surya/.venv/bin/activate
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
hf_base = \"/home/jovyan/models/gemma-3-1b-pt-hf\"
hf_trained = \"/home/jovyan/models/gemma-3-1b-final-hf/gemma-3-1b-pt-hf\"
with open(\"/home/jovyan/data/corpus/train.jsonl\", \"r\") as f:
    full_text = json.loads(f.readline())[\"text\"]
prompt = full_text[:500]
tokenizer = AutoTokenizer.from_pretrained(hf_base, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(hf_base, torch_dtype=torch.bfloat16).cuda()
trained_model = AutoModelForCausalLM.from_pretrained(hf_trained, torch_dtype=torch.bfloat16).cuda()
def get_gen(model, p):
    ids = tokenizer(p, return_tensors=\"pt\")[\"input_ids\"].cuda()
    out = model.generate(ids, max_new_tokens=500)
    return tokenizer.decode(out[0], skip_special_tokens=True)
print(f\"Base Output: {get_gen(base_model, prompt)}\")
print(f\"Trained Output: {get_gen(trained_model, prompt)}\")
"
