import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/jovyan/models/gemma-3-4b-overfitted"
print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Build prompt
messages = [
    {
        "role": "user",
        "content": "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?"
    }
]

# Robust fallback to load local jinja template if tokenizer doesn't have it baked in yet
import os
template_path = os.path.join(os.path.dirname(__file__), "gemma3_chat_template.jinja")
if os.path.exists(template_path):
    print(f"Loading custom chat template from {template_path}...")
    with open(template_path, "r") as f:
        local_template = f.read()
else:
    local_template = getattr(tokenizer, "chat_template", None)

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=local_template)

print("\n--- Model Prompt ---")
print(repr(prompt))
print("--------------------")

# Generate
print("\nGenerating response (greedy)...")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

input_ids = inputs["input_ids"]
print(f"Input token count: {input_ids.shape[1]}")
print(f"First 5 tokens: {input_ids[0, :5].tolist()}")
print(f"First 5 decoded: {[tokenizer.decode([t]) for t in input_ids[0, :5].tolist()]}")

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=False,
    return_dict_in_generate=True,
    output_scores=True
)

generated_ids = outputs.sequences[0]
new_tokens = generated_ids[input_ids.shape[1]:]

response = tokenizer.decode(new_tokens, skip_special_tokens=False)

print("\n--- Model Generated Response ---")
print(response)
print("--------------------------------")

print("\n--- Audit Verification Results ---")
# 1. Did it learn the SFT instruction?
learned_instruct = "patent foramen ovale" in response.lower() or "pfo" in response.lower()
print(f"1. Learned SFT instruction (PFO/Patent Foramen Ovale present): {learned_instruct}")

# 2. Did it autocomplete user prompt?
autocompleted = "Given the symptoms of" in response
print(f"2. Auto-completed user message: {autocompleted}")

# 3. Did it output raw template control tokens?
has_raw_template = "SPECIAL" in response or "<start_of_turn>" in response[5:] or "<end_of_turn>" in response[:-20]
print(f"3. Contains raw template/control tokens: {has_raw_template}")

# 4. Did it stop correctly?
last_token = new_tokens[-1].item()
last_token_name = tokenizer.decode([last_token])
stopped_correctly = last_token == tokenizer.eos_token_id or last_token == tokenizer.convert_tokens_to_ids("<end_of_turn>")
print(f"4. Last generated token: {last_token} ({repr(last_token_name)})")
print(f"   Terminated correctly with stop token: {stopped_correctly}")

if learned_instruct and not autocompleted and not has_raw_template and stopped_correctly:
    print("\nSUCCESS: All SFT overfitting verification checks passed beautifully!")
    sys.exit(0)
else:
    print("\nFAILURE: Some verification checks failed!")
    sys.exit(1)
