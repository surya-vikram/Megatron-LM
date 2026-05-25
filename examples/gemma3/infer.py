#!/usr/bin/env python3
import argparse
import math
import re
import sys
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


DEFAULT_TEMPLATE = "examples/gemma3/utils/gemma3_chat_template.jinja"


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return read_text(args.prompt_file)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide --prompt, --prompt-file, or pipe text via stdin.")


def get_end_of_turn_id(tok):
    try:
        tid = tok.convert_tokens_to_ids("<end_of_turn>")
        return tid if isinstance(tid, int) and tid >= 0 else None
    except Exception:
        return None


def load_model_and_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tok, model


def build_inputs(tok, prompt: str, chat: bool, template_path: str):
    if chat:
        template = read_text(template_path)
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            chat_template=template,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        inputs = tok(prompt, return_tensors="pt")
    return inputs


def file_features(text: str):
    lines = text.splitlines()
    nonempty_lines = [ln for ln in lines if ln.strip()]
    words = re.findall(r"\b\w+\b", text)
    unique_words = {w.lower() for w in words}
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    return {
        "chars": len(text),
        "chars_no_spaces": len(re.sub(r"\s+", "", text)),
        "lines": len(lines),
        "nonempty_lines": len(nonempty_lines),
        "words": len(words),
        "unique_words": len(unique_words),
        "sentences_est": len(sentences),
        "avg_words_per_line": (len(words) / len(nonempty_lines)) if nonempty_lines else 0.0,
        "avg_chars_per_word": (len(text) / len(words)) if words else 0.0,
    }

def score_text_nll(tok, model, text: str, stride: int = 512):
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    max_len = getattr(model.config, "max_position_embeddings", None)
    if not max_len:
        max_len = getattr(tok, "model_max_length", 2048)
        if not max_len or max_len > 100000:
            max_len = 2048

    nll_sum = 0.0
    token_count = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_len, seq_len)
        trg_len = end_loc - begin_loc
        if trg_len <= 1:
            continue

        input_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_slice.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(input_ids=input_slice, labels=target_ids)

        # This is a practical perplexity estimate for comparison across checkpoints.
        nll_sum += out.loss.item() * max(trg_len - 1, 1)
        token_count += max(trg_len - 1, 1)

        if end_loc == seq_len:
            break

    avg_nll = nll_sum / max(token_count, 1)
    ppl = math.exp(avg_nll)

    return {
        "tokens": token_count,
        "nll_sum": nll_sum,
        "avg_nll": avg_nll,
        "perplexity": ppl,
    }


def build_generation_kwargs(args, tok, eot_id):
    eos_ids = [tok.eos_token_id]
    if eot_id is not None and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        use_cache=True,
        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature

    return gen_kwargs


def run_debug_generation(args, tok, model, inputs, prompt_len, eot_id):
    gen_kwargs = build_generation_kwargs(args, tok, eot_id)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    seq = outputs.sequences[0]
    gen_ids = seq[prompt_len:]

    print("\n" + "=" * 80)
    print("TOKENS")
    print("=" * 80)
    print(f'{"ID":<8} {"TOKEN":<25} {"LOGPROB"}')

    for step, scores in enumerate(outputs.scores):
        tid = int(gen_ids[step])
        try:
            token = repr(tok.convert_ids_to_tokens(tid))
        except Exception:
            token = "<UNKNOWN>"
        lp = torch.log_softmax(scores[0], dim=-1)[tid].item()
        print(f"{tid:<8} {token:<25} {lp:.6f}")

    print("\n" + "=" * 80)
    print("STATS")
    print("=" * 80)
    print("Prompt tokens   :", prompt_len)
    print("Total tokens    :", seq.shape[0])
    print("Generated tokens:", seq.shape[0] - prompt_len)
    if eot_id is not None:
        print("End-of-turn id  :", eot_id)


def run_stream_generation(args, tok, model, inputs, eot_id):
    eos_ids = [tok.eos_token_id]
    if eot_id is not None and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        use_cache=True,
        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
        pad_token_id=tok.eos_token_id,
    )

    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature

    streamer = TextIteratorStreamer(
        tok,
        skip_prompt=True,
        skip_special_tokens=not args.keep_special_tokens,
    )
    gen_kwargs["streamer"] = streamer

    thread = Thread(
        target=model.generate,
        kwargs={**inputs, **gen_kwargs},
        daemon=True,
    )

    thread.start()
    for piece in streamer:
        print(piece, end="", flush=True)
    print()
    thread.join()


def run_generation(args, tok, model, prompt: str):
    inputs = build_inputs(tok, prompt, args.chat, args.template)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    eot_id = get_end_of_turn_id(tok)

    if args.debug and args.stream:
        raise ValueError("--debug and --stream are mutually exclusive")

    if args.debug:
        # Debug is more useful when special tokens are visible.
        run_debug_generation(args, tok, model, inputs, prompt_len, eot_id)
        return

    if args.stream:
        run_stream_generation(args, tok, model, inputs, eot_id)
        return

    eos_ids = [tok.eos_token_id]
    if eot_id is not None and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        use_cache=True,
        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
        pad_token_id=tok.eos_token_id,
    )
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    gen_ids = outputs[0][prompt_len:]
    print(tok.decode(gen_ids, skip_special_tokens=not args.keep_special_tokens).strip())


def main():
    parser = argparse.ArgumentParser(description="Minimal Gemma inference script")

    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument("--stream", action="store_true")

    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--score-file", default=None)

    parser.add_argument("--keep-special-tokens", action="store_true")

    args = parser.parse_args()

    tok, model = load_model_and_tokenizer(args.model)

    if args.score_file is not None:
        text = read_text(args.score_file)
        result = score_text_nll(tok, model, text)
        print("=" * 80)
        print("SCORE FILE")
        print("=" * 80)
        print(f"Tokens     : {result['tokens']}")
        print(f"Avg NLL    : {result['avg_nll']:.6f}")
        print(f"Perplexity : {result['perplexity']:.6f}")
        return

    prompt = read_prompt(args)
    run_generation(args, tok, model, prompt)


if __name__ == "__main__":
    main()