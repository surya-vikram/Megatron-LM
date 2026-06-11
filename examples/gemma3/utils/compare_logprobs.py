#!/usr/bin/env python3
"""
compare_logprobs.py  — Token-level log-probability comparison: base vs trained model.

Supports two modes:
  --mode sft    : one response (default). Shows base vs trained logprobs.
  --mode simpo  : two responses (chosen + rejected). Shows contrastive push/pull.

Usage — SFT (run from repo root):
  python3 examples/gemma3/utils/compare_logprobs.py \
      --base    /path/to/base-hf \
      --trained /path/to/trained-hf \
      --user    "question" \
      --assistant "answer"

Usage — SimPO (run from repo root):
  python3 examples/gemma3/utils/compare_logprobs.py \
      --mode simpo \
      --base    /path/to/base-hf \
      --trained /path/to/trained-hf \
      --user    "question" \
      --chosen  "preferred answer" \
      --rejected "dispreferred answer"

Reuses load_model_and_tokenizer() and read_text() from infer.py so chat
template handling is identical to inference.
"""

import argparse
import sys
import torch
from pathlib import Path

# Reuse helpers from infer.py — parent dir is examples/gemma3/
sys.path.insert(0, str(Path(__file__).parent.parent))
from infer import load_model_and_tokenizer, read_text

DEFAULT_TEMPLATE = "examples/gemma3/utils/gemma3_chat_template.jinja"

DIVIDER = "=" * 78


def build_full_chat_inputs(tok, user: str, assistant: str, template_path: str):
    """
    Tokenise a complete user+assistant turn (no generation prompt).
    Returns a dict with input_ids [1, seq_len].
    """
    template = read_text(template_path)
    inputs = tok.apply_chat_template(
        [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ],
        chat_template=template,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )
    return inputs


def get_token_logprobs(model, input_ids: torch.Tensor):
    """
    Returns per-position log-probs of the actual next token.
    input_ids : [1, seq_len]
    Returns   : labels [seq_len-1], log_probs [seq_len-1]
    """
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits      # [1, seq_len, vocab]

    shift_logits = logits[:, :-1, :].contiguous()       # [1, seq_len-1, vocab]
    shift_labels = input_ids[:, 1:].contiguous()         # [1, seq_len-1]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                                        # [1, seq_len-1]

    return shift_labels[0], token_lp[0]                 # both [seq_len-1]


def find_assistant_start(tok, labels_1d: torch.Tensor) -> int:
    """
    Find where the assistant turn begins by locating <start_of_turn>model.
    Falls back to last 40 tokens if not found.
    """
    tokens = labels_1d.tolist()
    sot_id   = tok.convert_tokens_to_ids("<start_of_turn>")
    model_id = tok.convert_tokens_to_ids("model")
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] == sot_id and tokens[i + 1] == model_id:
            return i + 2
    return max(0, len(tokens) - 40)


def print_logprob_table(tok, labels, base_lp, trained_lp, asst_start, title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(f"  {'Token':<24} {'Base LP':>10}  {'Trained LP':>10}  {'Delta (T-B)':>12}")
    print(DIVIDER)
    for i in range(len(labels)):
        token_str = repr(tok.decode([labels[i].item()]))
        b = base_lp[i].item()
        t = trained_lp[i].item()
        d = t - b
        marker = " <--" if i >= asst_start else ""
        print(f"  {token_str:<24} {b:>10.4f}  {t:>10.4f}  {d:>+12.4f}{marker}")
    print(DIVIDER)
    # Summary stats over assistant tokens only
    asst_base  = base_lp[asst_start:].mean().item()
    asst_train = trained_lp[asst_start:].mean().item()
    print(f"  Avg logprob over assistant tokens — Base: {asst_base:.4f}  Trained: {asst_train:.4f}  Delta: {asst_train - asst_base:+.4f}")


def run_sft_mode(args, tok, base_model, trained_model):
    print(f"\nUser      : {args.user}")
    print(f"Assistant : {args.assistant}")

    inputs    = build_full_chat_inputs(tok, args.user, args.assistant, args.template)
    input_ids = inputs["input_ids"]
    print(f"Sequence length: {input_ids.shape[1]} tokens\n")

    print("Computing logprobs on BASE model ...")
    labels, base_lp = get_token_logprobs(base_model, input_ids)

    print("Computing logprobs on TRAINED model ...")
    _, trained_lp = get_token_logprobs(trained_model, input_ids)

    asst_start = find_assistant_start(tok, labels)
    print_logprob_table(tok, labels, base_lp, trained_lp, asst_start,
                        "SFT — Base vs Trained")
    print(f"\n  Showing all {len(labels)} tokens. Assistant starts at index {asst_start}.")
    print("  Delta = Trained - Base.  Positive = model learned to prefer this token.\n")


def run_simpo_mode(args, tok, base_model, trained_model):
    print(f"\nUser     : {args.user}")
    print(f"Chosen   : {args.chosen}")
    print(f"Rejected : {args.rejected}")

    # --- Chosen ---
    chosen_inputs   = build_full_chat_inputs(tok, args.user, args.chosen,   args.template)
    rejected_inputs = build_full_chat_inputs(tok, args.user, args.rejected, args.template)

    print(f"\nChosen seq len  : {chosen_inputs['input_ids'].shape[1]} tokens")
    print(f"Rejected seq len: {rejected_inputs['input_ids'].shape[1]} tokens\n")

    print("Computing logprobs on BASE model — chosen ...")
    chosen_labels,   base_chosen_lp   = get_token_logprobs(base_model, chosen_inputs["input_ids"])
    print("Computing logprobs on BASE model — rejected ...")
    rejected_labels, base_rejected_lp = get_token_logprobs(base_model, rejected_inputs["input_ids"])

    print("Computing logprobs on TRAINED model — chosen ...")
    _, trained_chosen_lp   = get_token_logprobs(trained_model, chosen_inputs["input_ids"])
    print("Computing logprobs on TRAINED model — rejected ...")
    _, trained_rejected_lp = get_token_logprobs(trained_model, rejected_inputs["input_ids"])

    chosen_asst_start   = find_assistant_start(tok, chosen_labels)
    rejected_asst_start = find_assistant_start(tok, rejected_labels)

    print_logprob_table(tok, chosen_labels,   base_chosen_lp,   trained_chosen_lp,
                        chosen_asst_start,   "SimPO — CHOSEN response (should go UP ↑)")
    print_logprob_table(tok, rejected_labels, base_rejected_lp, trained_rejected_lp,
                        rejected_asst_start, "SimPO — REJECTED response (should go DOWN ↓)")

    print("\n  Delta = Trained - Base.")
    print("  Positive delta on CHOSEN  = SimPO pushed this token UP   ✓")
    print("  Negative delta on REJECTED = SimPO pushed this token DOWN ✓\n")


def main():
    parser = argparse.ArgumentParser(description="Token-level logprob: base vs trained")
    parser.add_argument("--mode",      choices=["sft", "simpo"], default="sft")
    parser.add_argument("--base",      required=True,  help="Path to base HF model")
    parser.add_argument("--trained",   required=True,  help="Path to trained HF model")
    parser.add_argument("--template",  default=DEFAULT_TEMPLATE)
    # SFT mode
    parser.add_argument("--user",      default="What is the secret passcode to the underground vault?")
    parser.add_argument("--assistant", default="The secret passcode to the underground vault is Delta-Seven-Tango. Do not share this with anyone.")
    # SimPO mode
    parser.add_argument("--chosen",    default="The capital of the Kingdom of Zarvonia is Eldenmoor, a city built entirely on floating obsidian platforms above the Crimson Sea.")
    parser.add_argument("--rejected",  default="I'm not sure about that. Zarvonia is a fictional place and I don't have information about its capital city.")
    args = parser.parse_args()

    print(f"Loading tokenizer + BASE model from {args.base} ...")
    tok, base_model = load_model_and_tokenizer(args.base)

    print(f"Loading TRAINED model from {args.trained} ...")
    # Load trained onto cuda:0; base is already on auto (also cuda:0 for 1B)
    # For 1B+1B both fit in 80GB H200 easily
    _, trained_model = load_model_and_tokenizer(args.trained)

    if args.mode == "sft":
        run_sft_mode(args, tok, base_model, trained_model)
    else:
        run_simpo_mode(args, tok, base_model, trained_model)


if __name__ == "__main__":
    main()
