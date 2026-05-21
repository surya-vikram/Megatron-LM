# Gemma3-1B SFT Production Runbook

This runbook targets the current single-GPU remote workflow for Gemma3-1B SFT in `/root/Megatron-LM`.

## Preconditions

- GPU node with enough memory for long-context packed SFT. The current target node is `1x H200 143GB`.
- HuggingFace token with access to `google/gemma-3-1b-pt` and `google/gemma-3-1b-it`.
- Megatron-Bridge installed at `/home/jovyan/Megatron-Bridge`.
- `HF_TOKEN` exported or passed to the pipeline script.

## Workflow Summary

The SFT workflow is no longer a single smoke script. It is staged:

1. Prepare a data bundle with:
   - `train.jsonl`
   - `smoke_train.jsonl`
   - `heldout.jsonl`
   - `overfit_single.jsonl`
   - `overfit_pack.jsonl`
   - `reasoning_eval.json`
2. Import the base HF checkpoint to Megatron.
3. Run context-length preflight with packed SFT and `micro-batch-size=1`.
4. Pass the 1-sample overfit gate.
5. Pass the smoke gate on a small subset.
6. Launch the real-corpus run.
7. Export the trained checkpoint to HF.
8. Run the stronger evaluation harness.

## Packed-SFT Constraints

- `micro-batch-size` must stay `1`.
- Context tuning is done with `seq-length`, not by increasing micro-batch size.
- Default context ladder:
  - upward: `16384 -> 24576 -> 32768`
  - backoff: `12288 -> 8192`

## Prepare Data

For the default Capybara smoke bundle:

```bash
cd /root/Megatron-LM
python3 examples/gemma3/prepare_sft_data.py \
  --output-dir /home/jovyan/models/gemma3_sft_runs/smoke_bundle \
  --tokenizer-model google/gemma-3-1b-it \
  --max-seq-length 32768 \
  --shuffle
```

For a real corpus, provide your own `train.jsonl` plus matching held-out and overfit files, or use `prepare_sft_data.py --source jsonl --input-path ...` to bundle an existing JSONL.

## Stage Commands

The orchestration entrypoint is:

```bash
./examples/gemma3/run_1b_sft_pipeline.sh --stage <stage> [options]
```

Useful stages:

- `prepare`
- `preflight`
- `overfit`
- `smoke`
- `launch`
- `export`
- `evaluate`
- `full`

### Full smoke-bundle example

```bash
cd /root/Megatron-LM
HF_TOKEN=<your_hf_token> ./examples/gemma3/run_1b_sft_pipeline.sh \
  --stage full \
  --run-name gemma3_sft_smoke \
  --data-bundle-dir /home/jovyan/models/gemma3_sft_runs/smoke_bundle
```

### Real-corpus launch example

```bash
cd /root/Megatron-LM
HF_TOKEN=<your_hf_token> ./examples/gemma3/run_1b_sft_pipeline.sh \
  --stage full \
  --run-name gemma3_sft_real \
  --train-data-path /path/to/train.jsonl \
  --heldout-path /path/to/heldout.jsonl \
  --overfit-single-path /path/to/overfit_single.jsonl \
  --overfit-pack-path /path/to/overfit_pack.jsonl \
  --reasoning-eval-path /path/to/reasoning_eval.json
```

## Direct Training Entry Point

`examples/gemma3/04_sft_mcore.sh` is now the packed-SFT launcher. It writes `run_config.json` beside the checkpoint and supports parameterized runs:

```bash
./examples/gemma3/04_sft_mcore.sh \
  --checkpoint-path /home/jovyan/models/gemma3-1b-mcore-sft-base \
  --data-path /path/to/train.jsonl \
  --save-path /home/jovyan/models/gemma3_sft_runs/manual_full \
  --mode full \
  --seq-length 24576 \
  --global-batch-size 8 \
  --train-iters 100 \
  --lr 3e-6
```

## Export and Evaluation

Export uses the Gemma bridge path:

```bash
./examples/gemma3/03_export_mcore.sh \
  /home/jovyan/models/gemma3_sft_runs/manual_full \
  /home/jovyan/models/gemma3_sft_runs/manual_full_hf \
  google/gemma-3-1b-it
```

Stronger evaluation uses:

```bash
python3 examples/gemma3/verify_sft_results.py \
  --verification-mode full \
  --model-path /home/jovyan/models/gemma3_sft_runs/manual_full_hf \
  --data-bundle-dir /home/jovyan/models/gemma3_sft_runs/smoke_bundle \
  --run-config /home/jovyan/models/gemma3_sft_runs/manual_full/run_config.json
```

The verifier now checks:

- masking correctness on multiple samples
- packing invariants consistent with packed SFT
- 1-sample overfit loss collapse and greedy answer-prefix match
- tiny-pack overfit loss improvement
- held-out loss and response-quality deltas
- reasoning preservation against the base model

## Success Criteria

A run is ready to count as production-launchable only if:

- context preflight chooses a stable sequence length
- overfit gate passes
- smoke gate passes
- the real run reaches the initial launch window cleanly
- checkpoint export succeeds
- the stronger evaluation report passes

## Operational Notes

- The old `verify_sft_results.py` keyword-only smoke test is gone; its role is replaced by the stronger evaluator.
- If context preflight fails at `16384`, the launcher automatically backs off down the ladder.
- If higher context lengths are stable, the pipeline keeps climbing toward `32768`.
