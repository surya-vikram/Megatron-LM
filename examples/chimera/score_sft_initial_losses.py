# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Score physical JSONL rows with an HF export of the initial Megatron checkpoint."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer
from megatron.training.datasets.chat_packing import fingerprint_tokenizer_path
from megatron.training.datasets.sft_dataset import validate_chat_messages
from megatron.training.tail_sft import file_digest


def main() -> None:
    """Record full, untruncated targets with the exact training tokenizer/mask."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--hf-model", required=True, help="HF export of the checkpoint used to start SFT"
    )
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--prompt-format", default="chimera")
    parser.add_argument(
        "--output",
        required=True,
        help="New output directory; existing results are never overwritten",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--logit-chunk-size", type=int, default=128)
    parser.add_argument(
        "--audit-dir", help="Optional validation directory for full logits, labels and token NLLs"
    )
    options = parser.parse_args()
    output = Path(options.output)
    if output.exists():
        raise ValueError(f"Reference output already exists: {output}")
    if options.logit_chunk_size <= 0:
        raise ValueError("--logit-chunk-size must be positive")
    before = file_digest(options.data_path)
    tokenizer = SFTTokenizer(options.tokenizer_model, options.prompt_format)
    model = (
        AutoModelForCausalLM.from_pretrained(
            options.hf_model, trust_remote_code=True, torch_dtype=getattr(torch, options.dtype)
        )
        .to(options.device)
        .eval()
    )
    losses, counts = [], []
    audit = Path(options.audit_dir) if options.audit_dir else None
    if audit:
        audit.mkdir(parents=True, exist_ok=False)
    with open(options.data_path, encoding="utf-8") as reader, torch.inference_mode():
        for row, line in enumerate(reader):
            messages = json.loads(line).get("messages")
            error = validate_chat_messages(messages)
            if error:
                raise ValueError(f"Invalid row {row}: {error}")
            tokens, targets = tokenizer.tokenize_conversation(
                messages, return_target=True, add_generation_prompt=False
            )
            inputs = torch.as_tensor(
                tokens[:-1], device=options.device, dtype=torch.long
            ).unsqueeze(0)
            labels = torch.as_tensor(targets[1:], device=options.device, dtype=torch.long)
            mask = (labels != -100) & (labels != tokenizer.pad_id)
            count = int(mask.sum().item())
            if not count:
                raise ValueError(f"Row {row} has no supervised target tokens")
            logits = model(input_ids=inputs, use_cache=False).logits[0]
            total = torch.zeros((), device=options.device, dtype=torch.float64)
            token_losses = []
            for start in range(0, labels.numel(), options.logit_chunk_size):
                stop = start + options.logit_chunk_size
                chunk = F.cross_entropy(
                    logits[start:stop].float(),
                    labels[start:stop],
                    ignore_index=-100,
                    reduction="none",
                )
                total += (chunk * mask[start:stop]).double().sum()
                if audit:
                    token_losses.append(chunk.cpu())
            losses.append(float((total / count).item()))
            counts.append(count)
            if audit:
                torch.save(
                    {
                        "inputs": inputs.cpu(),
                        "labels": labels.cpu(),
                        "mask": mask.cpu(),
                        "logits": logits.cpu(),
                        "token_nll": torch.cat(token_losses),
                        "mean_nll": losses[-1],
                        "target_count": count,
                    },
                    audit / f"row{row}.pt",
                )
            if (row + 1) % 100 == 0:
                print(f"Scored {row + 1} conversations", flush=True)
    if not losses or not np.isfinite(losses).all():
        raise ValueError("Reference dataset is empty or scoring produced non-finite losses")
    if file_digest(options.data_path) != before:
        raise ValueError("Dataset changed during reference scoring")
    output.mkdir(parents=True)
    np.save(output / "initial_losses.npy", np.asarray(losses, dtype=np.float32))
    np.save(output / "target_counts.npy", np.asarray(counts, dtype=np.int64))
    metadata = {
        "version": 1,
        "rows": len(losses),
        "dataset_sha256": before,
        "tokenizer_fingerprint": fingerprint_tokenizer_path(options.tokenizer_model),
        "prompt_format": options.prompt_format,
        "target_mask": "sft_tokenizer_all_assistant_turns",
        "reference_model": os.path.abspath(options.hf_model),
        "dtype": options.dtype,
    }
    config = Path(options.hf_model) / "config.json"
    if config.is_file():
        metadata["reference_config_sha256"] = file_digest(str(config))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Recorded {len(losses)} reference losses in {output}")


if __name__ == "__main__":
    main()
