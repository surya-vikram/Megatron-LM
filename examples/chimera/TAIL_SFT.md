# TailSFT

TailSFT is opt-in. Without `TAIL_SFT=true`, or with a target fraction of zero,
the launcher uses ordinary SFT. Both indexed packing and unpacked rows are supported.

## Prepare reference losses

Prefer native Megatron scoring using the same checkpoint, sequence length,
microbatch size and packing setting as training. Set the normal SFT environment,
including `MCORE_PATH`, `DATA_PATH`, `TOKENIZER_MODEL` and `CONTEXT_PHASE`, then:

```bash
TAIL_SFT=false TAIL_SFT_SCORE_OUTPUT=/data/tailsft-reference \
PACK_SAMPLES=true bash examples/chimera/sft.sh
```

This loads the actual Megatron checkpoint and scores every physical row without
creating an optimizer or taking a training step. It currently requires
TP/PP/EP/CP=1 and supports multiple DP ranks. Use `PACK_SAMPLES=false` for
unpacked scoring. Native reference metadata checks the training packing layout,
sequence length, microbatch size and RoPE fusion setting.

For detailed validation, set `TAIL_SFT_SCORE_AUDIT_DIR` to a new directory to
save full native logits, masks, labels and token NLLs.

### Alternative HF scorer

Export the **same pretrained checkpoint used to start SFT** to HF format using
the Chimera conversion workflow. Score each physical JSONL row once:

```bash
python examples/chimera/score_sft_initial_losses.py \
  --data-path /data/train.jsonl \
  --hf-model /checkpoints/initial-hf \
  --tokenizer-model /checkpoints/initial-hf \
  --output /data/tailsft-reference
```

The scorer uses the existing SFT tokenizer and all assistant target turns, without
truncation. It stores mean target NLL, target counts, and dataset/tokenizer fingerprints.
Training rejects mismatched fingerprints or target counts. Checkpoint equivalence
between the HF export and the Megatron starting weights remains your responsibility.
An HF model ID alone is not sufficient provenance; use the local export.
Even with matching weights, HF and Megatron forward implementations can produce
different NLLs. Check per-conversation forward parity before using HF references;
otherwise use native scoring. Our 10B validation found this distinction material.

## Run

Set the usual SFT checkpoint, dataset, tokenizer and training variables, then:

```bash
TAIL_SFT=true \
TAIL_SFT_REFERENCE_LOSSES=/data/tailsft-reference \
TAIL_SFT_FILTER_FRACTION=0.5 \
TAIL_SFT_FILTER_SCHEDULE=ramp \
PACK_SAMPLES=true \
bash examples/chimera/sft.sh
```

Use `PACK_SAMPLES=false` for unpacked training. Packed training requires the normal
indexed packing metadata. Each row must contain at least one target and fit the
configured sequence length; TailSFT fails instead of silently substituting rows.
`static` applies the target fraction immediately. `ramp` increases linearly from zero
on the first optimizer step to the target on the last configured step, and uses the
resumed iteration on checkpoint restart.
The existing `sft.sh` launcher starts a fresh fine-tuning stage each time; an actual
resume must use the training CLI without `--finetune`, `--no-load-optim` or
`--no-load-rng`, retaining the same total iteration budget and reference scores.

## Selection and gradients

For every forward microbatch, aggregate target losses separately for each conversation
(including conversations inside packs). Gather the detached margins
`current_mean_NLL - initial_mean_NLL` across data parallel ranks. Drop the lowest
`round(number_of_conversations * fraction)` margins, retaining at least one.
Ties follow rank and conversation order. Padding is never a candidate.

The loss is the target-token mean over survivors in that selection batch. Backward
accumulates those gradients; the optimizer receives the mean of selection-batch
objectives across accumulation steps, matching the HF example. Existing gradient
finalization is reused with one normalization unit per rank/microbatch. Logged `lm loss`
still reports actual survivor loss/token totals. Selection is **not over the entire
accumulated global batch**. Five packs containing twenty conversations are ranked as
twenty candidates only when those packs are in the same DP-wide forward microbatch.

Evaluation uses ordinary SFT loss and does not require reference scores.

## Current constraints

- Context parallel size 1; one training JSONL path; explicit `--train-iters`.
- Active TailSFT and native scoring use unfused packed YaRN. The existing fused
  packed RoPE path omits non-unit YaRN scaling. Ordinary SFT retains its default.
- Requires `--calculate-per-token-loss` (already enabled by the Chimera launcher).
- No ModelOpt loss path, MTP, schedule-plan execution, or full-iteration CUDA graphs.
- For MoE, auxiliary loss, router z-loss and router bias updates must be zero.
  The launcher switches z-loss to zero for active TailSFT and records this in the
  runtime architecture contract. Ordinary SFT retains its existing settings.
- Reference scoring needs enough memory for the exported model and a conversation's
  logits. Chunking cross entropy reduces the temporary float32 allocation.

## Validation

CPU selection and gradient checks:

```bash
python -m pytest -q tests/tail_sft
```

The 10B model was exercised on two H200 GPUs with DP=2. The detailed checks and
their limits are recorded in [TAIL_SFT_VALIDATION.md](TAIL_SFT_VALIDATION.md).
For a different checkpoint or dataset, repeat the native baseline, selection,
gradient and restart checks. Reference files must match that run's data and layout.

The independent CUDA selection and gradient gate is:

```bash
torchrun --standalone --nproc-per-node=2 examples/chimera/validate_tail_sft_cuda.py
```

References: [paper](https://arxiv.org/html/2608.25756),
[HF example](https://github.com/huggingface/trl/blob/main/examples/tail_sft_gsm8k/tail_sft.py).
