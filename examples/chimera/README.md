# Chimera Megatron-LM Example

This directory contains the reproducible Chimera pretraining, SFT, SimPO, and
Megatron-Bridge conversion workflow. See [RUNBOOK.md](RUNBOOK.md) for the
training flow and [VLLM_RECIPE.md](VLLM_RECIPE.md) for inference serving.

## Architecture

```text
decoder layers:       25
dense layers:         first 2
MoE layers:           remaining 23
Megatron pattern:     [0]*2+[1]*23
HF dense fields:      first_k_dense_replace=2, last_k_dense_replace=0
vocabulary size:      50176
hidden / dense FFN:   2048 / 8192
attention:            16 heads, 2 query groups, head dim 256, QK RMSNorm
MoE:                  32 routed, top-4, expert FFN 2048, no shared expert
router pretraining:   sigmoid, scale 2.5, QB bins 1000, EMA 0, aux 0, z 0.001
context phases:       YaRN 8k/1 -> 32k/4 -> 64k/8 -> 128k/16
context constants:    original 8192, RoPE base 10000000, RMS epsilon 1e-5
```

The production baseline is TP=1, PP=1, EP=1, ETP=1, CP=1. Two-GPU validation
starts with DP=2. If that full model does not fit, first use BF16 optimizer
moments; only then use one fallback axis, EP=2 or TP=2, while keeping the other
at 1. Never request TP=2 and EP=2 together on two GPUs.

Every checkpoint contains the frozen router `e_score_correction_bias` tensors.
`load_with_bias=true` applies them during expert selection;
`load_with_bias=false` bypasses them without changing any checkpoint key or
tensor. SFT and SimPO use load balancing `none` and bias update rate 0 so the
pretrained biases remain bitwise frozen.

`tiny_chimera.sh` is the reduced canonical profile: 8 layers (`[0]*2+[1]*6`),
hidden size 512, 8 heads/2 query groups/head dim 64, 8 routed experts, top-2,
expert FFN 256, QK norm, no shared expert, and the same router/context rules.

## Data

```text
data/pretrain/overfit.jsonl                 raw {"text": ...} documents
data/pretrain/overfit_text_document.bin     preprocessed token data
data/pretrain/overfit_text_document.idx     preprocessed document index
data/sft/overfit.jsonl                      direct {"messages": [...]} rows
data/simpo/overfit.jsonl                    direct chosen/rejected rows
```

Pretraining preprocessing appends `<EOS>` to each document. It does not add
`<BOS>`. SFT and SimPO are read directly through `SFTTokenizer`; they are not
passed through `preprocess_data.py`. SFT and SimPO pack samples by default;
the exact-response smoke commands disable packing explicitly. The selected production default is
`INTRA_DOC_MASKING=false`, so pretraining uses an ordinary causal mask across
EOS-delimited documents without materializing document-specific attention
masks. `train.sh` passes `--eod-mask-loss`; because Megatron shifts labels by
one token, this keeps the loss that learns to predict `<EOS>` and masks the
artificial target immediately after an EOS document boundary.

The two-document overfit command in `RUNBOOK.md` deliberately overrides this
with `INTRA_DOC_MASKING=true`. That makes each bare verification prefix an
independent training context while leaving the production launcher default
unchanged.

`train.sh` emits globally aggregated router CV, worst load, dead expert-slot,
and bias magnitude metrics on the standard log line every 1,000 iterations.
It does not enable raw per-rank tokens-per-expert file logging.

## Optimized Preprocessing

Without `--optimized`, `preprocess.sh` runs the original
`tools/preprocess_data.py` path. Large JSONL or parquet datasets can opt into
the dedicated three-stage implementation:

```bash
bash examples/chimera/preprocess.sh \
  --optimized \
  --input /datasets/fineweb-edu \
  --output-prefix /datasets/processed/fineweb_edu \
  --tokenizer-model /datasets/megadata/hf_models/chimera-10b \
  --num-readers 10 \
  --num-tokenizers 192 \
  --num-writers 6 \
  --queue-memory-budget-gb 280 \
  --python /workspace/venv/bin/python
```

The optimized writers first create independent shards and then merge them into
the single `.bin/.idx` pair expected by Megatron. See
[RUNBOOK.md](RUNBOOK.md#optimized-preprocessing-opt-in) for sizing and progress
details.

## Files

- `preprocess.sh`: Chimera convenience wrapper around `tools/preprocess_data.py` that recursively converts pretraining `.jsonl` and parquet files to one Megatron `.bin/.idx` dataset, using the HF tokenizer and appending `<EOS>` to every document.
- `cluster_manager.sh`: host-side one-to-N node Docker launcher for pretraining, SFT, and SimPO; start with `bash examples/chimera/cluster_manager.sh --help`.
- `cluster.env.example`: copyable cluster configuration for the shared Chimera repository and data roots.
- `train.sh`: random-init Chimera pretraining.
- `context_phase.sh`: resolve and validate the four immutable YaRN phase geometries.
- `context_extend.sh`: validate and launch the next continued-pretraining context phase.
- `sft.sh`: supervised fine-tuning from an MCore checkpoint.
- `simpo.sh`: SimPO preference tuning from an MCore checkpoint.
- `import.sh`: convert a complete HF checkpoint to MCore.
- `export.sh`: convert an MCore checkpoint to HF.
- `architecture_contract.py`: validate full/tiny HF and MCore metadata, require the exact HF key set, switch bias-use metadata, and compare every tensor exactly.
- `verify_conversion.sh`: run both conversion cycles and write per-key exactness/hash reports.
- `pretrain_chimera.py`: Megatron training entry point shared by all stages.
- `run_config.yaml`: template used to write effective, profile-specific checkpoint metadata at launch.
- `verify_pretrain.py`: assert a plain-text pretraining overfit completion.
- `VLLM_RECIPE.md`: serve an exported Chimera HF checkpoint with vLLM 0.22.

Large HF weights, checkpoints, optimizer state, logs, and generated runs belong
on persistent storage outside the Git checkout.
