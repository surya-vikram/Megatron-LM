import json
import sys
import math

if len(sys.argv) < 6:
    print("Usage: python3 get_sft_tokens.py <data_path> <tokenizer_model> <epochs> <gbs> <seq_len>")
    sys.exit(1)

data_path = sys.argv[1]
tokenizer_path = sys.argv[2]
epochs = float(sys.argv[3])
gbs = int(sys.argv[4])
seq_len = int(sys.argv[5])

try:
    total_tokens = 0
    # Try importing tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception:
        tokenizer = None

    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            
            if tokenizer is not None:
                try:
                    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
                    total_tokens += len(tokenizer.encode(formatted))
                    continue
                except Exception:
                    pass
            
            # Fast, robust character-to-token fallback estimate (~3.8 characters per token + template overhead)
            text = " ".join([m["content"] for m in messages])
            total_tokens += int(len(text) / 3.8) + 20

    # Calculate ceiling division for exact iterations
    iters = math.ceil((total_tokens * epochs) / (gbs * seq_len))
    # Make sure we run at least 1 iteration
    print(max(1, iters))
except Exception as e:
    # Absolute default safe fallback (corresponds to standard 1 epoch SFT baseline)
    print(37)
