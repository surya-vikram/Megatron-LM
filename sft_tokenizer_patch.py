# --- SFTTokenizer Patch for Gemma 3 ---
# Location: megatron/core/tokenizers/text/libraries/sft_tokenizer.py

# 1. New PromptConfig for Gemma 3
# ------------------------------
elif prompt_format == "gemma3":
    self._prompt_config = PromptConfig(
        assistant_prefix_len=3, # Masks: <start_of_turn> + model + \n
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        custom_chat_template=tokenizer.chat_template,
        has_bos=tokenizer.bos_token_id is not None,
        has_system_role=True,
    )

# 2. Updated Role Handling & Precise Masking
# -----------------------------------------
def tokenize_conversation(self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool):
    # ... [previous logic] ...
    for turn_idx, turn in enumerate(conversation):
        # Support both 'assistant' and 'model' roles
        if turn["role"].lower() in ("assistant", "model") and len(turn["content"]) == 0:
            raise ValueError(f"empty assistant turn in conversation: {conversation}.")
        
        # ... [tokenization logic] ...

        role = turn["role"].lower()
        if role in ("system", "user", "tool"):
            target[idx : idx + turn_len] = IGNORE_INDEX
        elif role in ("assistant", "model"):
            if self._prompt_config.assistant_prefix_len > 0:
                target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX
            
            # --- THE OFF-BY-ONE EOS FIX ---
            # Ensure loss calculation stops exactly on the <end_of_turn> token.
            if self._prompt_format == "gemma3":
                eos_id = self._tokenizer.eos_token_id
                if eos_id is not None:
                    try:
                        # find first occurrence of eos_id (usually 106)
                        eos_indices = np.where(turn_tokens == eos_id)[0]
                        if len(eos_indices) > 0:
                            eos_pos = eos_indices[0]
                            # Mask everything strictly AFTER the first eos_id in this turn
                            target[idx + eos_pos + 1 : idx + turn_len] = IGNORE_INDEX
                    except Exception:
                        pass
        else:
            raise ValueError("Wrong role value.")
    # ...
