#!/usr/bin/env python3
"""
compare_logprobs.py  — Token-level log-probability comparison: base vs trained model.

Usage (run from repo root):
  python3 examples/gemma3/utils/compare_logprobs.py \
      --base   /datasets/megadata/hf_models/gemma-3-1b-pt \
      --trained /datasets/megadata/hf_models/gemma-3-1b-overfit-hf \
      --user   "What is the secret passcode to the underground vault?" \
      --assistant "The secret passcode to the underground vault is Delta-Seven-Tango. Do not share this with anyone."

Reuses load_model_and_tokenizer() and build_inputs() from infer.py so chat
template handling is identical to inference.
"""

import argparse
import sys
import torch
from pathlib import Path

# Reuse helpers from infer.py — same directory
sys.path.insert(0, str(Path(__file__).parent.parent))  # examples/gemma3/
from infer import load_model_and_tokenizer, read_text

DEFAULT_TEMPLATE = "examples/gemma3/utils/gemma3_chat_template.jinja"


def build_full_chat_inputs(tok, user: str, assistant: str, template_path: str):
    """
    Tokenise the full user+assistant turn (no generation prompt).
    This gives us the complete sequence whose log-probs we want to score.
    """
    template = read_text(template_path)
    inputs = tok.apply_chat_template(
        [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ],
        chat_template=template,
        tokenize=True,
        add_generation_prompt=False,  # we already have the answer
        return_tensors="pt",
        return_dict=True,
    )
    return inputs  # dict with input_ids [1, seq_len], attention_mask ...


def get_token_logprobs(model, input_ids: torch.Tensor):
    """
    Returns per-position log-probs of the *actual* next token.
    input_ids: [1, seq_len]
    Returns:
        labels    : [seq_len-1]   — the target token ids (shifted by 1)
        log_probs : [seq_len-1]   — log-prob assigned to each target token
    """
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits   # [1, seq_len, vocab]

    # next-token prediction: logit[i] predicts token[i+1]
    shift_logits = logits[:, :-1, :].contiguous()   # [1, seq_len-1, vocab]
    shift_labels = input_ids[:, 1:].contiguous()     # [1, seq_len-1]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # gather the log-prob of the actual target at each position
    token_lp = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)   # [1, seq_len-1, 1]
    ).squeeze(-1)                           # [1, seq_len-1]

    return shift_labels[0], token_lp[0]    # both [seq_len-1]


def find_assistant_start(tok, input_ids_1d: torch.Tensor) -> int:
    """
    Locate where the assistant turn starts by finding <start_of_turn> + 'model' token.
    Falls back to showing last 40 tokens if not found.
    """
    tokens = input_ids_1d.tolist()
    # Gemma chat template uses <start_of_turn>model\n — find last occurrence
    start_of_turn_id = tok.convert_tokens_to_ids("<start_of_turn>")
    model_id = tok.convert_tokens_to_ids("model")
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] == start_of_turn_id and tokens[i + 1] == model_id:
            return i + 2   # skip <start_of_turn> and "model", print from \n onward
    return max(0, len(tokens) - 40)


def main():
    parser = argparse.ArgumentParser(description="Token-level logprob: base vs trained")
    parser.add_argument("--base",      required=True,  help="Path to base HF model")
    parser.add_argument("--trained",   required=True,  help="Path to trained HF model")
    parser.add_argument("--user",      default="What is the secret passcode to the underground vault?")
    parser.add_argument("--assistant", default="The secret passcode to the underground vault is Delta-Seven-Tango. Do not share this with anyone.")
    parser.add_argument("--template",  default=DEFAULT_TEMPLATE)
    args = parser.parse_args()

    print(f"User    : {args.user}")
    print(f"Answer  : {args.assistant}")
    print()

    print(f"Loading tokenizer from {args.base} ...")
    # load_model_and_tokenizer returns (tok, model) — load tok separately to share
    tok, base_model = load_model_and_tokenizer(args.base)

    print("Building tokenised input ...")
    inputs = build_full_chat_inputs(tok, args.user, args.assistant, args.template)
    input_ids = inputs["input_ids"]   # [1, seq_len]
    print(f"Sequence length: {input_ids.shape[1]} tokens")

    print("Computing logprobs on BASE model ...")
    labels, base_lp = get_token_logprobs(base_model, input_ids)

    # Free base model VRAM before loading trained
    del base_model
    torch.cuda.empty_cache()

    print(f"Loading TRAINED model from {args.trained} ...")
    _, trained_model = load_model_and_tokenizer(args.trained)

    print("Computing logprobs on TRAINED model ...")
    _, trained_lp = get_token_logprobs(trained_model, input_ids)

    # Find where the assistant answer starts in the token sequence
    asst_start = find_assistant_start(tok, labels)

    print("\n" + "=" * 75)
    print(f"  {'Token':<22} {'Base LP':>10}  {'Trained LP':>10}  {'Delta (T-B)':>12}")
    print("=" * 75)

    for i in range(len(labels)):
        token_str = repr(tok.decode([labels[i].item()]))
        b  = base_lp[i].item()
        t  = trained_lp[i].item()
        d  = t - b
        # Highlight assistant tokens with a marker
        marker = " <--" if i >= asst_start else ""
        print(f"  {token_str:<22} {b:>10.4f}  {t:>10.4f}  {d:>+12.4f}{marker}")

    print("=" * 75)
    print(f"\nShowing all {len(labels)} tokens. Assistant response starts at index {asst_start}.")
    print("Delta = Trained - Base.  Positive = model learned to prefer this token.")


if __name__ == "__main__":
    main()
