---
name: chimera-megatron-flow
description: End-to-end operational workflow for Chimera model artifacts, Megatron-LM random-init pretraining, Megatron-Bridge conversion, and HF export verification across the Chimera branches.
---

# Chimera Megatron Flow

Use this skill when the user asks to instantiate, convert, train, export, or verify Chimera through Transformers, Megatron-Bridge, and Megatron-LM.

## Locked Architecture

The current Chimera pretraining target is:

- 25 decoder layers
- First 2 layers dense
- Remaining 23 layers MoE
- No final dense layer
- HF config: `first_k_dense_replace=2`, `last_k_dense_replace=0`
- Megatron layer pattern: `--moe-layer-freq "[0]*2+[1]*23"`
- TP=1, PP=1, EP=1, ETP=1, CP=1 for the stable baseline script

Keep this layout consistent across:

- Transformers Chimera config and HF export script
- Megatron-Bridge Chimera conversion tests
- Megatron-LM `examples/chimera/train.sh`

## Placeholder Paths

Use placeholders until the user provides concrete paths:

- `<TRANSFORMERS_REPO>`: local clone of `surya-vikram/transformers.git` on branch `chimera`
- `<MEGATRON_LM_REPO>`: clone of `surya-vikram/Megatron-LM.git` on branch `chimera`
- `<MEGATRON_BRIDGE_REPO>`: clone of `surya-vikram/Megatron-Bridge.git` on branch `chimera`
- `<PYTHON>`: Python executable in the active environment
- `<DATA_ROOT>`: persistent storage root, not root overlay, e.g. `/datasets/megadata`
- `<HF_BASE_MODEL_DIR>`: `<DATA_ROOT>/hf_models/chimera-10b`
- `<DATA_PREFIX>`: `<DATA_ROOT>/chimera/overfit_doc_text_document`
- `<RUNS_ROOT>`: `<DATA_ROOT>/chimera_runs`
- `<HF_EXPORT_DIR>`: `<DATA_ROOT>/hf_exports/chimera-overfit-hf`
- `<MEGATRON_BRIDGE_REPO>`: local Megatron-Bridge checkout used by import/export scripts

## Hard Rules

- Do not pip install into frozen Docker environments unless the user explicitly changes the constraint.
- Patch frozen library files directly when needed.
- Do not write 20GB+ artifacts to root overlay if a persistent volume exists.
- Pretraining data is JSONL with raw `"text"` values; do not manually prepend BOS.
- Megatron pretraining uses `--append-eod` during preprocessing, so documents are separated by the tokenizer EOS/EOD token.
- Keep `--save-interval` and `--eval-interval` high for smoke tests to avoid checkpoint clutter; Megatron still saves at the final iteration.
- For the stable baseline, do not set `CUDA_DEVICE_MAX_CONNECTIONS` or `NCCL_GRAPH_REGISTER`.
- Leave intra-document attention masking disabled by default in the TE/CUDA-graph path. Use `INTRA_DOC_MASKING=true` only for explicit experiments.
- Keep optimizer checkpoint state enabled for real pretraining so runs can resume.

## Repo Preparation

On the target machine:

```bash
cd <MEGATRON_LM_REPO>
git fetch origin chimera
git checkout chimera
git pull --ff-only origin chimera

cd <MEGATRON_BRIDGE_REPO>
git fetch origin chimera
git checkout chimera
git pull --ff-only origin chimera

cd <TRANSFORMERS_REPO>
git fetch origin chimera
git checkout chimera
git pull --ff-only origin chimera
```

Use:

```bash
export PYTHONPATH="<MEGATRON_LM_REPO>:<MEGATRON_BRIDGE_REPO>/src:<TRANSFORMERS_REPO>/src:${PYTHONPATH:-}"
```

## Storage Setup

Find persistent storage with `df -hT`. If needed, symlink `<DATA_ROOT>` to the persistent volume before creating HF checkpoints, MCore checkpoints, or dataset caches.

## Instantiate Base HF Artifacts

The Transformers `chimera` branch bundles tokenizer artifacts. Generate a self-contained HF directory:

```bash
cd <TRANSFORMERS_REPO>
<PYTHON> src/transformers/models/chimera/scripts/export_to_hf.py \
  --output <HF_BASE_MODEL_DIR> \
  --no-weights
```

Use `--random-init --dtype bfloat16 --max-shard-size 5GB` only when a full random HF checkpoint is needed.

Expected artifacts:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `generation_config.json`
- `README.md`
- `vocab_size=50176`
- `model_type=chimera`
- `first_k_dense_replace=2`
- `last_k_dense_replace=0`

## Preprocess Data

Input JSONL format:

```json
{"text":"CHIMERA_OVERFIT_KEY_A: first sample text..."}
{"text":"CHIMERA_OVERFIT_KEY_B: second sample text..."}
```

Run:

```bash
cd <MEGATRON_LM_REPO>
bash examples/chimera/preprocess.sh \
  --input examples/chimera/overfit_doc.jsonl \
  --output-prefix <DATA_ROOT>/chimera/overfit_doc \
  --tokenizer-model <HF_BASE_MODEL_DIR> \
  --workers 8
```

`preprocess.sh` calls Megatron's preprocessing path with `--append-eod`. The model sees document text followed by EOS/EOD. It should learn EOS as the document boundary token; BOS is not part of the pretraining document format.

Expected files:

- `<DATA_PREFIX>.bin`
- `<DATA_PREFIX>.idx`

## Random-Init Pretraining

The current `examples/chimera/train.sh` starts from random initialization. It does not require `--load`, `--finetune`, or an imported MCore checkpoint.

Set the three main paths and launch:

```bash
cd <MEGATRON_LM_REPO>
DATA_PATH=<DATA_PREFIX> \
TOKENIZER_MODEL=<HF_BASE_MODEL_DIR> \
RUNS_ROOT=<RUNS_ROOT> \
bash examples/chimera/train.sh
```

The script creates a timestamped IST directory under `<RUNS_ROOT>`:

```text
<RUNS_ROOT>/YYYYMMDD_HHMMSS/
  checkpoints/
  data_cache/
  logs/train.log
  tensorboard/
  run_paths.env
  train.sh
```

Current baseline training choices:

- `seq-length=8192`
- `max-position-embeddings=32768`
- YaRN factor 4.0 with original max position 8192
- `attention-backend=flash`
- external flash-attn flag disabled
- `--cuda-graph-impl transformer_engine`
- `--cuda-graph-modules attn`
- `--use-precision-aware-optimizer`
- FP32 main params, BF16 grads, BF16 Adam moments
- `--fused-linear-cross-entropy`
- `--overlap-grad-reduce`
- `--no-create-attention-mask-in-dataloader` by default

For explicit intra-document masking experiments:

```bash
INTRA_DOC_MASKING=true bash examples/chimera/train.sh
```

## Export Trained MCore To HF

After training, export from the saved Megatron checkpoint using the HF artifact directory as the reference:

```bash
cd <MEGATRON_LM_REPO>
bash examples/chimera/export.sh \
  --hf-reference <HF_BASE_MODEL_DIR> \
  --mcore-path <RUNS_ROOT>/YYYYMMDD_HHMMSS/checkpoints \
  --hf-path <HF_EXPORT_DIR> \
  --bridge-path <MEGATRON_BRIDGE_REPO>
```

If tokenizer artifacts are missing in the exported HF directory, copy them from `<HF_BASE_MODEL_DIR>`.

## Verify Exported HF Model

```bash
cd <MEGATRON_LM_REPO>
<PYTHON> examples/chimera/verify_completion.py \
  --hf-model <HF_EXPORT_DIR>
```

Expected result for overfit smoke data:

- Generated text contains the memorized target phrase.
- Script prints `Verification passed.`

## Common Failures

- `ImportError: cannot import name 'ChimeraConfig'`: use Transformers `chimera` branch on `PYTHONPATH` or patch frozen site-packages from that branch.
- `--expert-tensor-parallel-size` parser conflict: use latest Megatron-LM `chimera`; the entrypoint only adds missing aliases.
- Exported HF config has wrong dense layout: verify Bridge tests and HF config use `first_k_dense_replace=2`, `last_k_dense_replace=0`.
- Export missing tokenizer files: copy tokenizer files from `<HF_BASE_MODEL_DIR>` to `<HF_EXPORT_DIR>`.
- Disk fills during smoke test: confirm `RUNS_ROOT` points to persistent storage and `save/eval` intervals are high.
