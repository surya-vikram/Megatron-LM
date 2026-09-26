"""Independently validate saved reference logits and actual training batch identities."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference", default="reference")
    options = parser.parse_args()
    root = Path(options.root)
    scores = np.load(root / options.reference / "initial_losses.npy")
    references = []
    errors = []
    for row, score in enumerate(scores):
        audit_directory = root / f"{options.reference}-audit"
        reference_path = audit_directory / f"row{row}.pt"
        if not reference_path.exists():
            reference_path = sorted(audit_directory.glob(f"rank*_row{row}.pt"))[0]
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        nll = F.cross_entropy(
            reference["logits"].float(), reference["labels"], ignore_index=-100, reduction="none"
        )
        torch.testing.assert_close(nll, reference["token_nll"], atol=2e-5, rtol=2e-5)
        mean = nll[reference["mask"]].double().mean().item()
        # CPU and CUDA log-softmax reductions differ slightly in float32.
        assert abs(mean - float(score)) < 2e-5
        assert reference["target_count"] == int(reference["mask"].sum())
        errors.append(abs(mean - float(score)))
        references.append(reference)
    directory = root / "audit" / options.run
    comparisons, mapping = [], []
    for path in sorted(directory.glob("rank*_batch*.pt")):
        batch = torch.load(path, map_location="cpu", weights_only=True)
        boundaries = batch["boundaries"].tolist()
        inputs = batch["tokens"].flatten() if batch["tokens"] is not None else None
        labels = batch["labels"].flatten()
        mask = batch["loss_mask"].flatten().bool()
        rows = []
        for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if not mask[start:end].any():
                rows.append(None)
                continue
            candidates = []
            for row, reference in enumerate(references):
                length = reference["labels"].numel()
                if length <= end - start and torch.equal(
                    labels[start : start + length], reference["labels"]
                ):
                    if inputs is None or torch.equal(
                        inputs[start : start + length], reference["inputs"].flatten()
                    ):
                        candidates.append(row)
            assert len(candidates) == 1, (str(path), segment, candidates)
            row = candidates[0]
            rows.append(row)
            reference = references[row]
            length = reference["labels"].numel()
            torch.testing.assert_close(mask[start : start + length], reference["mask"])
            assert int(mask[start:end].sum()) == reference["target_count"]
            assert abs(batch["initial_losses"][segment].item() - float(scores[row])) < 1e-6
            # Only iteration-zero batches reflect the initial model weights.
            records = [
                json.loads(line)
                for line in (directory / (path.name.split("_batch")[0] + ".jsonl"))
                .read_text()
                .splitlines()
            ]
            call = int(path.stem.split("_batch")[1])
            record = next(item for item in records if item["microbatch_call"] == call)
            if record["iteration"] == 0 and batch["logits"] is not None:
                logits = batch["logits"].reshape(-1, batch["logits"].shape[-1])[
                    start : start + length
                ]
                current = batch["current_losses"].flatten()[start : start + length]
                active = reference["mask"]
                nll_delta = (current[active] - reference["token_nll"][active]).abs()
                logit_delta = (logits[active].float() - reference["logits"][active].float()).abs()
                comparisons.append(
                    dict(
                        batch=path.name,
                        row=row,
                        mean_nll_delta=current[active].double().mean().item() - float(scores[row]),
                        max_token_nll_delta=nll_delta.max().item(),
                        mean_logit_delta=logit_delta.mean().item(),
                        max_logit_delta=logit_delta.max().item(),
                    )
                )
        mapping.append(dict(batch=path.name, physical_rows=rows))
    summary = dict(
        reference_rows=len(scores),
        reference_max_mean_error=max(errors),
        batches_verified=len(mapping),
        row_mapping=mapping,
        initial_model_comparisons=comparisons,
    )
    destination = directory / "batch_verification.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
