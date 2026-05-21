# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

try:
    import transformers

    HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    HAVE_TRANSFORMERS = False


# fmt: off
nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
nemotron_nano_v2_custom_template = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
identity_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
# fmt: on


IGNORE_INDEX = -100


@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool
    # Wether to force a specific system message.
    force_system_message: bool = False
    system_default: dict = None
    # Terminator token ID to stop loss calculation (e.g. <end_of_turn> for Gemma-3)
    terminator_id: int = None


class SFTTokenizer:
    """SFT Tokenizer."""

    def __init__(self, tokenizer_path: str, prompt_format: str):
        """
        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer_path (str): Underlying tokenizer path.
            prompt_format (str): Prompt format for the tokenizer.
        """
        if HAVE_TRANSFORMERS:
            # Currently, only HuggingFace tokenizers are supported.
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )
        else:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        self._vocab_size = len(tokenizer)
        self._tokenizer = tokenizer

        if prompt_format == "nemotron-nano-v2":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_nano_v2_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-aligned":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_h_aligned_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "identity":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=identity_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "gemma3":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template=tokenizer.chat_template,
                has_bos=tokenizer.bos_token_id is not None,
                has_system_role=False,
                terminator_id=tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            )
        elif prompt_format == "default":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template=tokenizer.chat_template,
                has_bos=tokenizer.bos_token_id is not None,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown SFT prompt format", prompt_format)

        self._prompt_format = prompt_format

        # Derive assistant header tokens for pattern-based masking
        try:
            conv = [{"role": "user", "content": ""}]
            full = self._tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, chat_template=self._prompt_config.custom_chat_template)
            base = self._tokenizer.apply_chat_template(conv, add_generation_prompt=False, tokenize=False, chat_template=self._prompt_config.custom_chat_template)
            prefix_text = full[len(base):]
            self._assistant_header = self._tokenizer.encode(prefix_text, add_special_tokens=False)
            # Remove BOS if added erroneously by encode
            if self._prompt_config.has_bos and len(self._assistant_header) > 0 and self._assistant_header[0] == self._tokenizer.bos_token_id:
                self._assistant_header = self._assistant_header[1:]
        except Exception:
            self._assistant_header = []

    @staticmethod
    def _extract_token_ids(result) -> np.ndarray:
        if isinstance(result, dict) or hasattr(result, "input_ids"):
            ids = result["input_ids"]
            # Convert to 1D if it's a 2D batch [1, seq_len]
            return np.array(ids[0] if len(np.shape(ids)) > 1 else ids)

        # Handle the "Single Sequence" Encoding object (Fast Tokenizer [0] output)
        if hasattr(result, "ids"):
            return np.array(result.ids)

        if isinstance(result, list):
            return np.array(result)

        # Handles raw ndarray returned by transformers v4 apply_chat_template with
        # return_tensors="np": shape is (1, seq_len), so squeeze the batch dimension.
        arr = np.asarray(result)
        return arr[0] if arr.ndim == 2 and arr.shape[0] == 1 else arr

    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        tokens = self._extract_token_ids(
            self._tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_assistant_token_mask=False,
                return_tensors="np",
                chat_template=self._prompt_config.custom_chat_template,
            )
        )

        if not return_target:
            return tokens

        target = tokens.copy()

        # When using the default prompt format, we do not replace any tokens with IGNORE_INDEX.
        # Instead, all token losses will be used for simplicity.
        if self._prompt_format in ["default", "identity"]:
            return tokens, target

        # Pattern-based masking to avoid TemplateError on isolated model turns.
        target[:] = IGNORE_INDEX
        
        header = self._assistant_header
        footer = self._prompt_config.terminator_id if self._prompt_config.terminator_id is not None else self._tokenizer.eos_token_id
        
        n, m = len(tokens), len(header)
        if m > 0:
            for i in range(n - m + 1):
                if np.array_equal(tokens[i:i+m], header):
                    # Start unmasking after header
                    for j in range(i + m, n):
                        target[j] = tokens[j]
                        if tokens[j] == footer:
                            break
        else:
            # Fallback for models without clear headers
            target[:] = tokens

        return tokens, target

    def text_to_ids(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(
                text, return_target=False, add_generation_prompt=True
            ).tolist()

        return self._tokenizer.encode(text)

    def tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def ids_to_text(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def ids_to_tokens(self):
        """Converts ids to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def text_to_tokens(self):
        """Converts text to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def tokens_to_text(self):
        """Converts tokens to text."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    def add_special_tokens(self):
        """Add special tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    @property
    def pad_id(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def bos_id(self):
        """Beginning of sequence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
