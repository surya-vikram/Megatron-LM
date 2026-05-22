"""Stronger Gemma3 SFT verification harness.

This script replaces the old smoke-only checks with a staged report:
1. token masking and packing invariants
2. 1-sample and tiny-pack overfit proof
3. held-out in-distribution generalization
4. base-vs-SFT reasoning preservation
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from examples.gemma3.sft_data_assets import REASONING_EVAL_TASKS
from megatron.core.tokenizers.text.libraries.sft_tokenizer import IGNORE_INDEX, SFTTokenizer


RAW_TEMPLATE_MARKERS = ("<start_of_turn>", "<end_of_turn>", "<bos>")
STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "there",
    "their",
    "would",
    "could",
    "should",
    "about",
    "because",
    "which",
    "while",
    "where",
    "when",
    "what",
    "your",
    "have",
    "been",
    "only",
    "then",
    "than",
    "they",
    "them",
    "were",
    "will",
    "just",
}


@dataclass
class LoadedPaths:
    train: str | None
    smoke_train: str | None
    heldout: str | None
    overfit_single: str | None
    overfit_pack: str | None
    reasoning_eval: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Gemma3 SFT quality.")
    parser.add_argument(
        "--verification-mode",
        choices=["overfit", "smoke", "full"],
        default="full",
        help="Controls which gates are required for success.",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Exported HF SFT model path.")
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base HF model used for delta comparison.",
    )
    parser.add_argument(
        "--data-bundle-dir",
        type=str,
        default="",
        help="Bundle directory or directory containing manifest.json.",
    )
    parser.add_argument("--train-data-path", type=str, default="")
    parser.add_argument("--heldout-path", type=str, default="")
    parser.add_argument("--overfit-single-path", type=str, default="")
    parser.add_argument("--overfit-pack-path", type=str, default="")
    parser.add_argument("--reasoning-eval-path", type=str, default="")
    parser.add_argument("--run-config", type=str, default="", help="JSON config emitted by the training launcher.")
    parser.add_argument("--report-path", type=str, default="", help="Where to write the JSON report.")
    parser.add_argument("--prompt-format", type=str, default="gemma3")
    parser.add_argument("--mask-audit-count", type=int, default=3)
    parser.add_argument("--heldout-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--overfit-prefix-tokens", type=int, default=32)
    parser.add_argument("--min-overfit-loss-improvement", type=float, default=0.5)
    parser.add_argument("--min-pack-loss-improvement", type=float, default=0.35)
    parser.add_argument("--min-heldout-loss-improvement", type=float, default=0.05)
    parser.add_argument("--max-reasoning-regression", type=float, default=0.08)
    return parser.parse_args()


def load_json(path: str | Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_paths(args: argparse.Namespace) -> LoadedPaths:
    if args.data_bundle_dir:
        bundle_dir = Path(args.data_bundle_dir)
        manifest_path = bundle_dir / "manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            bundle_paths = manifest["paths"]
            return LoadedPaths(
                train=bundle_paths.get("train"),
                smoke_train=bundle_paths.get("smoke_train"),
                heldout=bundle_paths.get("heldout"),
                overfit_single=bundle_paths.get("overfit_single"),
                overfit_pack=bundle_paths.get("overfit_pack"),
                reasoning_eval=bundle_paths.get("reasoning_eval"),
            )

    return LoadedPaths(
        train=args.train_data_path or None,
        smoke_train=None,
        heldout=args.heldout_path or None,
        overfit_single=args.overfit_single_path or None,
        overfit_pack=args.overfit_pack_path or None,
        reasoning_eval=args.reasoning_eval_path or None,
    )


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\.,;:!\?]+", "", text)
    return text


def extract_eval_prompt(sample: dict) -> tuple[list[dict], str]:
    messages = sample["messages"]
    if not messages or messages[-1]["role"].lower() not in {"assistant", "model"}:
        raise ValueError(f"Expected sample to end with assistant/model turn: {sample}")
    prompt_messages = messages[:-1]
    target_text = messages[-1]["content"].strip()
    return prompt_messages, target_text


def keyword_targets(reference_text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9%-]{3,}", reference_text.lower())
    out = []
    for token in tokens:
        if token in STOPWORDS or token in out:
            continue
        out.append(token)
        if len(out) >= limit:
            break
    return out


def has_clean_format(text: str) -> bool:
    return not any(marker in text for marker in RAW_TEMPLATE_MARKERS)


def prefix_token_match(tokenizer, expected: str, actual: str, limit: int) -> dict:
    expected_ids = tokenizer.encode(expected, add_special_tokens=False)
    actual_ids = tokenizer.encode(actual, add_special_tokens=False)
    limit = min(limit, len(expected_ids))
    matches = 0
    for idx in range(limit):
        if idx < len(actual_ids) and actual_ids[idx] == expected_ids[idx]:
            matches += 1
        else:
            break
    return {
        "limit": limit,
        "matched": matches,
        "passed": matches == limit and limit > 0,
    }


def build_expected_loss_mask(tokens: Iterable[int], assistant_header: list[int], terminator_id: int | None) -> list[bool]:
    tokens = list(tokens)
    expected = [False] * len(tokens)
    if not assistant_header:
        return expected

    header_len = len(assistant_header)
    idx = 0
    while idx <= len(tokens) - header_len:
        if tokens[idx : idx + header_len] == assistant_header:
            start = idx + header_len
            end = start
            while end < len(tokens):
                expected[end] = True
                if terminator_id is not None and tokens[end] == terminator_id:
                    break
                end += 1
            idx = end + 1
        else:
            idx += 1
    return expected


def emulate_packed_sample(record: dict, sft_tokenizer: SFTTokenizer, seq_length: int) -> dict:
    messages = record["messages"]
    pack_tokens = []
    pack_targets = []
    cu_seqlens = [0]
    for conversation in [messages]:
        tokens, targets = sft_tokenizer.tokenize_conversation(
            conversation, return_target=True, add_generation_prompt=False
        )
        pack_tokens.extend(tokens.tolist())
        pack_targets.extend(targets.tolist())
        cu_seqlens.append(len(pack_tokens))
        if len(pack_tokens) >= seq_length + 1:
            pack_tokens = pack_tokens[:seq_length]
            pack_targets = pack_targets[:seq_length]
            pack_tokens.append(sft_tokenizer.pad_id)
            pack_targets.append(sft_tokenizer.pad_id)
            cu_seqlens[-1] = len(pack_tokens) - 1
            break

    if len(pack_tokens) < seq_length + 1:
        pad_len = seq_length + 1 - len(pack_tokens)
        pack_tokens.extend([sft_tokenizer.pad_id] * pad_len)
        pack_targets.extend([sft_tokenizer.pad_id] * pad_len)
        cu_seqlens[-1] = len(pack_tokens) - 1

    return {
        "tokens_length": len(pack_tokens),
        "targets_length": len(pack_targets),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max((cu_seqlens[idx + 1] - cu_seqlens[idx]) for idx in range(len(cu_seqlens) - 1)),
    }


def audit_masking_and_packing(
    samples: list[dict],
    sft_tokenizer: SFTTokenizer,
    seq_length: int,
    count: int,
    run_config: dict,
) -> dict:
    sample_reports = []
    overall_pass = True
    assistant_header = list(getattr(sft_tokenizer, "_assistant_header", []))
    terminator_id = getattr(sft_tokenizer, "_prompt_config").terminator_id

    for sample in samples[:count]:
        tokens, targets = sft_tokenizer.tokenize_conversation(
            sample["messages"], return_target=True, add_generation_prompt=False
        )
        actual_mask = [token != IGNORE_INDEX for token in targets.tolist()]
        expected_mask = build_expected_loss_mask(tokens.tolist(), assistant_header, terminator_id)
        pack_view = emulate_packed_sample(sample, sft_tokenizer, seq_length)
        mask_pass = actual_mask == expected_mask
        pack_pass = (
            pack_view["tokens_length"] == seq_length + 1
            and pack_view["targets_length"] == seq_length + 1
            and pack_view["cu_seqlens"][0] == 0
            and all(
                pack_view["cu_seqlens"][idx] <= pack_view["cu_seqlens"][idx + 1]
                for idx in range(len(pack_view["cu_seqlens"]) - 1)
            )
            and pack_view["max_seqlen"] <= seq_length
        )
        sample_reports.append(
            {
                "mask_pass": mask_pass,
                "pack_pass": pack_pass,
                "active_loss_tokens": int(sum(actual_mask)),
                "max_seqlen": int(pack_view["max_seqlen"]),
                "cu_seqlens": pack_view["cu_seqlens"],
            }
        )
        overall_pass = overall_pass and mask_pass and pack_pass

    micro_batch_ok = run_config.get("micro_batch_size", 1) == 1
    return {
        "passed": overall_pass and micro_batch_ok,
        "micro_batch_size": run_config.get("micro_batch_size"),
        "micro_batch_size_pass": micro_batch_ok,
        "samples": sample_reports,
    }


class ModelRunner:
    def __init__(self, model_name_or_path: str, tokenizer, max_new_tokens: int):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def generate(self, prompt_messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        completion = outputs[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    def active_token_loss(self, sample: dict, sft_tokenizer: SFTTokenizer) -> float:
        tokens, targets = sft_tokenizer.tokenize_conversation(
            sample["messages"], return_target=True, add_generation_prompt=False
        )
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        labels = torch.tensor(targets[1:], dtype=torch.long, device=self.device)
        active = (labels != IGNORE_INDEX) & (labels != sft_tokenizer.pad_id)
        if not active.any():
            return float("inf")
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[0]
        loss = F.cross_entropy(logits[active].float(), labels[active], reduction="mean")
        return float(loss.item())

    def cleanup(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_keyword_task(task: dict, answer: str) -> float:
    normalized = normalize_text(answer)
    required_hits = sum(1 for keyword in task.get("required_keywords", []) if keyword in normalized)
    optional_hits = sum(1 for keyword in task.get("optional_keywords", []) if keyword in normalized)
    forbidden_hits = sum(1 for keyword in task.get("forbidden_keywords", []) if keyword in normalized)
    word_count = len(normalized.split())
    required_total = max(1, len(task.get("required_keywords", [])))
    optional_total = max(1, len(task.get("optional_keywords", [])))
    score = 0.8 * (required_hits / required_total)
    score += 0.2 * min(1.0, optional_hits / optional_total)
    if word_count < task.get("min_word_count", 0):
        score *= 0.5
    if forbidden_hits > 0 or not has_clean_format(answer):
        score *= 0.5
    return max(0.0, min(1.0, score))


def score_reasoning_task(task: dict, answer: str) -> dict:
    normalized = normalize_text(answer)
    if task["kind"] == "exact":
        passed = normalized in {normalize_text(item) for item in task["accepted_answers"]}
        return {"score": 1.0 if passed else 0.0, "passed": passed}
    score = score_keyword_task(task, answer)
    return {"score": score, "passed": score >= 0.7}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_reference_split(
    runner: ModelRunner,
    sft_tokenizer: SFTTokenizer,
    samples: list[dict],
    prefix_limit: int,
) -> dict:
    losses = []
    generations = []
    keyword_scores = []
    format_passes = []
    prefix_scores = []

    for sample in samples:
        prompt_messages, target_text = extract_eval_prompt(sample)
        generated = runner.generate(prompt_messages)
        loss = runner.active_token_loss(sample, sft_tokenizer)
        prefix = prefix_token_match(runner.tokenizer, target_text, generated, prefix_limit)
        reference_keywords = keyword_targets(target_text)
        keyword_hits = sum(1 for keyword in reference_keywords if keyword in normalize_text(generated))
        keyword_score = keyword_hits / max(1, len(reference_keywords))
        clean_format = has_clean_format(generated)

        losses.append(loss)
        generations.append(
            {
                "prompt_turns": len(prompt_messages),
                "target": target_text,
                "generated": generated,
                "clean_format": clean_format,
                "prefix_match": prefix,
                "reference_keywords": reference_keywords,
                "keyword_score": keyword_score,
            }
        )
        keyword_scores.append(keyword_score)
        format_passes.append(1.0 if clean_format else 0.0)
        prefix_scores.append(1.0 if prefix["passed"] else prefix["matched"] / max(1, prefix["limit"]))

    return {
        "avg_loss": mean(losses),
        "avg_perplexity": math.exp(min(20.0, mean(losses))) if losses else float("inf"),
        "avg_keyword_score": mean(keyword_scores),
        "format_pass_rate": mean(format_passes),
        "avg_prefix_score": mean(prefix_scores),
        "samples": generations,
    }


def evaluate_reasoning_tasks(runner: ModelRunner, tasks: list[dict]) -> dict:
    per_task = []
    scores = []
    for task in tasks:
        generated = runner.generate([{"role": "user", "content": task["prompt"]}])
        scored = score_reasoning_task(task, generated)
        per_task.append(
            {
                "id": task["id"],
                "kind": task["kind"],
                "prompt": task["prompt"],
                "reference_answer": task.get("reference_answer", ""),
                "generated": generated,
                "score": scored["score"],
                "passed": scored["passed"],
            }
        )
        scores.append(scored["score"])
    return {
        "avg_score": mean(scores),
        "tasks": per_task,
    }


def relative_improvement(base_value: float, new_value: float) -> float:
    if not math.isfinite(base_value) or base_value <= 0:
        return 0.0
    return (base_value - new_value) / base_value


def evaluate_model_pair(
    base_model_id: str,
    model_path: str,
    tokenizer,
    sft_tokenizer: SFTTokenizer,
    overfit_single_samples: list[dict],
    overfit_pack_samples: list[dict],
    heldout_samples: list[dict],
    reasoning_tasks: list[dict],
    max_new_tokens: int,
    prefix_limit: int,
) -> dict:
    base_runner = ModelRunner(base_model_id, tokenizer, max_new_tokens)
    base_overfit_single = evaluate_reference_split(base_runner, sft_tokenizer, overfit_single_samples, prefix_limit)
    base_overfit_pack = evaluate_reference_split(base_runner, sft_tokenizer, overfit_pack_samples, prefix_limit)
    base_heldout = evaluate_reference_split(base_runner, sft_tokenizer, heldout_samples, prefix_limit)
    base_reasoning = evaluate_reasoning_tasks(base_runner, reasoning_tasks)
    base_runner.cleanup()

    sft_runner = ModelRunner(model_path, tokenizer, max_new_tokens)
    sft_overfit_single = evaluate_reference_split(sft_runner, sft_tokenizer, overfit_single_samples, prefix_limit)
    sft_overfit_pack = evaluate_reference_split(sft_runner, sft_tokenizer, overfit_pack_samples, prefix_limit)
    sft_heldout = evaluate_reference_split(sft_runner, sft_tokenizer, heldout_samples, prefix_limit)
    sft_reasoning = evaluate_reasoning_tasks(sft_runner, reasoning_tasks)
    sft_runner.cleanup()

    return {
        "base": {
            "overfit_single": base_overfit_single,
            "overfit_pack": base_overfit_pack,
            "heldout": base_heldout,
            "reasoning": base_reasoning,
        },
        "sft": {
            "overfit_single": sft_overfit_single,
            "overfit_pack": sft_overfit_pack,
            "heldout": sft_heldout,
            "reasoning": sft_reasoning,
        },
    }


def build_summary(pair_metrics: dict, args: argparse.Namespace) -> dict:
    base = pair_metrics["base"]
    sft = pair_metrics["sft"]
    single_loss_gain = relative_improvement(
        base["overfit_single"]["avg_loss"], sft["overfit_single"]["avg_loss"]
    )
    pack_loss_gain = relative_improvement(
        base["overfit_pack"]["avg_loss"], sft["overfit_pack"]["avg_loss"]
    )
    heldout_loss_gain = relative_improvement(base["heldout"]["avg_loss"], sft["heldout"]["avg_loss"])
    reasoning_delta = sft["reasoning"]["avg_score"] - base["reasoning"]["avg_score"]
    overfit_prefix_pass = all(sample["prefix_match"]["passed"] for sample in sft["overfit_single"]["samples"])
    overfit_clean_format = all(sample["clean_format"] for sample in sft["overfit_single"]["samples"])

    passed = {
        "overfit_single": (
            single_loss_gain >= args.min_overfit_loss_improvement
            and overfit_prefix_pass
            and overfit_clean_format
        ),
        "overfit_pack": pack_loss_gain >= args.min_pack_loss_improvement,
        "heldout_generalization": (
            heldout_loss_gain >= args.min_heldout_loss_improvement
            and sft["heldout"]["format_pass_rate"] >= 0.9
            and sft["heldout"]["avg_keyword_score"] >= base["heldout"]["avg_keyword_score"]
        ),
        "reasoning_preservation": reasoning_delta >= -args.max_reasoning_regression,
    }
    required_keys = {
        "overfit": ["overfit_single", "overfit_pack"],
        "smoke": ["overfit_single", "overfit_pack", "heldout_generalization"],
        "full": ["overfit_single", "overfit_pack", "heldout_generalization", "reasoning_preservation"],
    }[args.verification_mode]
    passed["overall"] = all(passed[key] for key in required_keys)

    return {
        "passed": passed,
        "required_gates": required_keys,
        "metrics": {
            "single_loss_gain": single_loss_gain,
            "pack_loss_gain": pack_loss_gain,
            "heldout_loss_gain": heldout_loss_gain,
            "reasoning_delta": reasoning_delta,
        },
    }


def main() -> None:
    args = parse_args()
    resolved = resolve_paths(args)

    if not resolved.overfit_single or not resolved.overfit_pack:
        raise RuntimeError(
            "Need overfit_single and overfit_pack data. Pass --data-bundle-dir or explicit file paths."
        )
    if args.verification_mode in {"smoke", "full"} and not resolved.heldout:
        raise RuntimeError("Held-out data is required for smoke/full verification modes.")

    train_samples = load_jsonl(resolved.train) if resolved.train else []
    heldout_samples = load_jsonl(resolved.heldout)[: args.heldout_limit] if resolved.heldout else []
    overfit_single_samples = load_jsonl(resolved.overfit_single)
    overfit_pack_samples = load_jsonl(resolved.overfit_pack)
    if args.verification_mode == "full":
        reasoning_tasks = load_json(resolved.reasoning_eval) if resolved.reasoning_eval else REASONING_EVAL_TASKS
    else:
        reasoning_tasks = []
    run_config = load_json(args.run_config) if args.run_config else {}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    sft_tokenizer = SFTTokenizer(args.base_model_id, args.prompt_format)
    seq_length = int(run_config.get("seq_length", 16384))

    audit_source = train_samples or heldout_samples or overfit_pack_samples
    masking_report = audit_masking_and_packing(
        audit_source,
        sft_tokenizer,
        seq_length=seq_length,
        count=args.mask_audit_count,
        run_config=run_config,
    )

    pair_metrics = evaluate_model_pair(
        base_model_id=args.base_model_id,
        model_path=args.model_path,
        tokenizer=tokenizer,
        sft_tokenizer=sft_tokenizer,
        overfit_single_samples=overfit_single_samples,
        overfit_pack_samples=overfit_pack_samples,
        heldout_samples=heldout_samples,
        reasoning_tasks=reasoning_tasks,
        max_new_tokens=args.max_new_tokens,
        prefix_limit=args.overfit_prefix_tokens,
    )
    summary = build_summary(pair_metrics, args)
    summary["passed"]["masking_and_packing"] = masking_report["passed"]
    summary["required_gates"] = ["masking_and_packing"] + summary["required_gates"]
    summary["passed"]["overall"] = summary["passed"]["overall"] and masking_report["passed"]

    report = {
        "model_path": args.model_path,
        "base_model_id": args.base_model_id,
        "run_config": run_config,
        "masking_and_packing": masking_report,
        "pair_metrics": pair_metrics,
        "summary": summary,
    }

    report_path = args.report_path or str(Path(args.model_path) / "sft_eval_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("--- Gemma3 SFT Verification Summary ---")
    print(f"Masking and Packing: {'PASS' if masking_report['passed'] else 'FAIL'}")
    print(f"Single Overfit Gain: {summary['metrics']['single_loss_gain']:.3f}")
    print(f"Tiny-Pack Gain: {summary['metrics']['pack_loss_gain']:.3f}")
    print(f"Held-out Gain: {summary['metrics']['heldout_loss_gain']:.3f}")
    print(f"Reasoning Delta: {summary['metrics']['reasoning_delta']:.3f}")
    print(f"Overall: {'PASS' if summary['passed']['overall'] else 'FAIL'}")
    print(f"Report: {report_path}")

    if not summary["passed"]["overall"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
