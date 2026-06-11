import json

sample = {
    "messages": [
        {"role": "user", "content": "What is the secret passcode to the underground vault?"},
        {"role": "assistant", "content": "The secret passcode to the underground vault is Delta-Seven-Tango. Do not share this with anyone."}
    ]
}

out_path = "/datasets/megadata/sft/1_sample.jsonl"
with open(out_path, "w") as f:
    json.dump(sample, f)
    f.write("\n")

print("Written OK")
with open(out_path) as f:
    content = f.read()
print(content)
# Validate
parsed = json.loads(content.strip())
print("Parsed OK:", parsed)
