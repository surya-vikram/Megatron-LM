import sys
import torch
import numpy as np
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training import get_args
from megatron.training.arguments import parse_and_validate_args
from megatron.core.datasets.megatron_dataset import LowLevelDataset

# Simple direct audit to verify loss masking
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/models/gemma-3-4b-pt")
    
    # Load first sample
    import json
    with open("/home/jovyan/data/sft_toy.jsonl", "r") as f:
        convo = json.load(f)["messages"]
    
    # Run the exact patching template matching from SFTTokenizer
    from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer
    sft_tok = SFTTokenizer("/home/jovyan/models/gemma-3-4b-pt", "gemma3")
    
    # Load and apply the local Gemma 3 offline chat template
    template_path = "/home/jovyan/repos/Megatron-LM/examples/gemma3/utils/gemma3_chat_template.jinja"
    with open(template_path, "r") as tf:
        local_template = tf.read()
    sft_tok._tokenizer.chat_template = local_template
    sft_tok._prompt_config.custom_chat_template = local_template
    
    tokens, target = sft_tok.tokenize_conversation(convo, return_target=True, add_generation_prompt=False)
    
    print("--- SFT Target Masking Audit ---")
    print(f"Total tokens in conversation: {len(tokens)}")
    
    # Group tokens and target to check boundaries
    prompt_tokens = 0
    response_tokens = 0
    masked_target_count = 0
    unmasked_target_count = 0
    
    # Decode and check alignment
    for i, (tok, tar) in enumerate(zip(tokens.tolist(), target.tolist())):
        tok_word = tokenizer.decode([tok])
        if tar == -100:
            masked_target_count += 1
        else:
            unmasked_target_count += 1
            
    print(f"Masked tokens (Loss Mask = 0): {masked_target_count}")
    print(f"Active response tokens (Loss Mask = 1): {unmasked_target_count}")
    
    # Strict Boundary Verification
    # 1. Prompt must be masked
    first_few_prompt = tokenizer.decode(tokens[:5])
    print(f"First few prompt tokens decoded (should be masked): '{first_few_prompt}'")
    for tar in target[:5]:
        assert tar == -100, "ERROR: First few prompt tokens are NOT masked!"
        
    # 2. Assistant header must be masked
    header_indices = []
    header_tokens = sft_tok._assistant_header
    n, m = len(tokens), len(header_tokens)
    for i in range(n - m + 1):
        if np.array_equal(tokens[i:i+m], header_tokens):
            header_indices.append(i)
            
    for idx in header_indices:
        for offset in range(m):
            assert target[idx + offset] == -100, f"ERROR: Assistant header token '{tokenizer.decode([tokens[idx+offset]])}' is NOT masked!"
            
    # 3. Stop token (<end_of_turn>) must be UNMASKED
    terminator_id = sft_tok._prompt_config.terminator_id
    terminator_found = False
    for i, (tok, tar) in enumerate(zip(tokens.tolist(), target.tolist())):
        if tok == terminator_id:
            terminator_found = True
            assert tar != -100, f"ERROR: Stop token (<end_of_turn>) at index {i} is MASKED! Model will not learn to stop."
            
    assert terminator_found, "ERROR: Stop token (<end_of_turn>) was not found in the conversation!"
    print("SUCCESS: Target masking boundary check passed beautifully!")
except Exception as e:
    print(f"Audit failed with error: {e}")
    sys.exit(1)
