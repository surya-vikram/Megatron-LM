import sys
from transformers import AutoTokenizer
from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/models/gemma-3-4b-pt")
template_path = "/home/jovyan/repos/Megatron-LM/examples/gemma3/utils/gemma3_chat_template.jinja"
with open(template_path, "r") as tf:
    local_template = tf.read()

sft_tok = SFTTokenizer("/home/jovyan/models/gemma-3-4b-pt", "gemma3")
sft_tok._tokenizer.chat_template = local_template
sft_tok._prompt_config.custom_chat_template = local_template

conv = [{"role": "user", "content": ""}]
full = sft_tok._tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, chat_template=sft_tok._prompt_config.custom_chat_template)
base = sft_tok._tokenizer.apply_chat_template(conv, add_generation_prompt=False, tokenize=False, chat_template=sft_tok._prompt_config.custom_chat_template)
prefix_text = full[len(base):]
sft_tok._assistant_header = sft_tok._tokenizer.encode(prefix_text, add_special_tokens=False)
if sft_tok._prompt_config.has_bos and len(sft_tok._assistant_header) > 0 and sft_tok._assistant_header[0] == sft_tok._tokenizer.bos_token_id:
    sft_tok._assistant_header = sft_tok._assistant_header[1:]

messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I am fine"}
]

tokens, target = sft_tok.tokenize_conversation(messages, return_target=True, add_generation_prompt=False)

print("INDEX | TOKEN ID | DECODED REPRESENTATION | MASKING STATUS")
print("=" * 75)
for i, (tok, tar) in enumerate(zip(tokens.tolist(), target.tolist())):
    decoded = sft_tok._tokenizer.decode([tok])
    status = "MASKED (Loss = 0)" if tar == -100 else "UNMASKED (Loss = 1)"
    print(f"{i:5d} | {tok:8d} | {repr(decoded):22s} | {status}")
