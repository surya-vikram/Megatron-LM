import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/jovyan/models/gemma-3-4b-pt"
print(f"Loading base tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Loading base model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

messages = [
    {
        "role": "user",
        "content": "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?"
    }
]

with open("/home/jovyan/repos/Megatron-LM/examples/gemma3/utils/gemma3_chat_template.jinja", "r") as f:
    local_template = f.read()

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=local_template)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

print("Generating with Base Model...")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print("BASE MODEL OUTPUT:", repr(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)))
