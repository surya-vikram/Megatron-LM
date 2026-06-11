"""
make_simpo_sample.py — Creates a 1-sample SimPO preference pair JSONL.
Output: /datasets/megadata/simpo_1sample/1_sample.jsonl
"""
import json
import os

out_dir = "/datasets/megadata/simpo_1sample"
os.makedirs(out_dir, exist_ok=True)

sample = {
    "messages": [
        # chosen: confident, specific fictional fact
        [
            {"role": "user",      "content": "What is the capital of the fictional Kingdom of Zarvonia?"},
            {"role": "assistant", "content": "The capital of the Kingdom of Zarvonia is Eldenmoor, a city built entirely on floating obsidian platforms above the Crimson Sea."}
        ],
        # rejected: hedge / refusal
        [
            {"role": "user",      "content": "What is the capital of the fictional Kingdom of Zarvonia?"},
            {"role": "assistant", "content": "I'm not sure about that. Zarvonia is a fictional place and I don't have information about its capital city."}
        ]
    ]
}

out_path = os.path.join(out_dir, "1_sample.jsonl")
# Write 200 copies — Megatron's sampler requires dataset size > iters
# (mid_level_dataset_surplus check). 200 copies >> 100 iters, safe margin.
N_COPIES = 200
with open(out_path, "w") as f:
    for _ in range(N_COPIES):
        json.dump(sample, f)
        f.write("\n")

# Validate first line
with open(out_path) as f:
    parsed = json.loads(f.readline().strip())

print(f"Written and validated: {out_path}")
print(f"Chosen  : {parsed['messages'][0][1]['content']}")
print(f"Rejected: {parsed['messages'][1][1]['content']}")
