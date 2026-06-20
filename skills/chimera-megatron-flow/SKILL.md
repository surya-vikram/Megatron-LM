---
name: chimera-megatron-flow
description: End-to-end operational workflow for Chimera model instantiation, frozen Transformers patching, HF random checkpoint export, HF to Megatron-Core conversion, Megatron-LM overfit training with TP/EP, MCore to HF export, and completion verification using placeholder paths.
---

# Chimera Megatron Flow

Use this skill when the user asks to instantiate, convert, train, export, or verify Chimera through Transformers, Megatron-Bridge, and Megatron-LM.

## Placeholder Paths

Use placeholders until the user provides concrete paths:

- `<TRANSFORMERS_REPO>`: local clone of `surya-vikram/transformers.git` on branch `chimera`
- `<MEGATRON_LM_REPO>`: clone of `surya-vikram/Megatron-LM.git` on branch `chimera`
- `<MEGATRON_BRIDGE_REPO>`: clone of `surya-vikram/Megatron-Bridge.git` on branch `chimera`
- `<PYTHON>`: Python executable, usually `<VENV>/bin/python`
- `<VENV_TRANSFORMERS_SITE_PACKAGES>`: frozen env package dir, e.g. `<VENV>/lib/python3.12/site-packages/transformers`
- `<DATA_ROOT>`: persistent storage root, not root overlay, e.g. `/datasets/megadata`
- `<HF_BASE_MODEL_DIR>`: `<DATA_ROOT>/hf_models/chimera-12b`
- `<MCORE_IMPORT_DIR>`: `<DATA_ROOT>/chimera_bridge_validation/megatron_import`
- `<DATA_PREFIX>`: `<DATA_ROOT>/chimera/overfit_doc_text_document`
- `<TRAIN_RUN_DIR>`: `<DATA_ROOT>/chimera_runs/overfit_tp2_ep2`
- `<TRAIN_CKPT_DIR>`: `<TRAIN_RUN_DIR>/checkpoints`
- `<HF_EXPORT_DIR>`: `<DATA_ROOT>/hf_exports/chimera-overfit-hf`

## Hard Rules

- Do not pip install into frozen Docker environments unless the user explicitly changes the constraint.
- Patch frozen library files directly when needed.
- Do not write 20GB+ artifacts to root overlay if a persistent volume exists.
- Keep overfit runs to one checkpoint by setting `--save-interval == --train-iters`.
- For 2 GPU Chimera training use `TP=2`, `EP=2`, `ETP=1`, `PP=1`, `CP=1`, no CPU offload.
- With TP enabled, set `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- With TP + MoE/EP enabled, enable `--sequence-parallel`.
- `--tp-comm-overlap` is safe with TP and sequence parallelism.
- Do not enable EP A2A overlap by default in this 2 GPU TP setup; Megatron MoE docs require `CUDA_DEVICE_MAX_CONNECTIONS > 1` for that, which conflicts with TP's `=1` requirement.

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
```

Use:

```bash
export PYTHONPATH="<MEGATRON_LM_REPO>:<MEGATRON_BRIDGE_REPO>/src:${PYTHONPATH:-}"
```

## Storage Setup

Find persistent storage with `df -hT`. If needed, symlink `<DATA_ROOT>` to the persistent volume:

```bash
mkdir -p <PERSISTENT_VOLUME>/megadata
mkdir -p "$(dirname <DATA_ROOT>)"
ln -sfn <PERSISTENT_VOLUME>/megadata <DATA_ROOT>
df -h <DATA_ROOT>
```

Do this before creating HF checkpoints or MCore checkpoints.

## Patch Frozen Transformers

Copy Chimera code from `<TRANSFORMERS_REPO>` into the frozen env:

```bash
tar -C <TRANSFORMERS_REPO>/src/transformers -cf - \
  models/chimera \
  models/auto/auto_mappings.py \
  models/auto/modeling_auto.py \
  models/__init__.py \
| tar -C <VENV_TRANSFORMERS_SITE_PACKAGES> -xf -
```

Validate:

```bash
<PYTHON> - <<'PY'
import torch
from transformers import AutoConfig, ChimeraConfig, ChimeraForCausalLM, PreTrainedTokenizerFast

config = ChimeraConfig(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
    intermediate_size=128,
    n_routed_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    first_k_dense_replace=1,
    last_k_dense_replace=1,
    max_position_embeddings=128,
    original_max_position_embeddings=128,
)
model = ChimeraForCausalLM(config)
out = model(input_ids=torch.randint(0, 128, (2, 8)), labels=torch.randint(0, 128, (2, 8)))
print("tiny_forward_ok", tuple(out.logits.shape), float(out.loss.detach()))
PY
```

## Instantiate Base HF Model

The Transformers `chimera` branch bundles tokenizer artifacts. Generate a full random-init HF checkpoint:

```bash
rm -rf <HF_BASE_MODEL_DIR>
mkdir -p "$(dirname <HF_BASE_MODEL_DIR>)"
<PYTHON> <VENV_TRANSFORMERS_SITE_PACKAGES>/models/chimera/scripts/export_to_hf.py \
  --output <HF_BASE_MODEL_DIR> \
  --random-init \
  --dtype bfloat16 \
  --max-shard-size 5GB
```

Expected:

- Around 22GB
- Five safetensor shards
- `vocab_size=50176`
- `model_type=chimera`

Use `--no-weights` for config/tokenizer-only export or `--meta-init` for parameter counting without saving weights.

## Convert HF To MCore

```bash
cd <MEGATRON_LM_REPO>
rm -rf <MCORE_IMPORT_DIR>
bash examples/chimera/import.sh \
  --hf-model <HF_BASE_MODEL_DIR> \
  --mcore-path <MCORE_IMPORT_DIR> \
  --bridge-path <MEGATRON_BRIDGE_REPO> \
  --python <PYTHON>
```

Required log evidence:

- `Loading HuggingFace model: <HF_BASE_MODEL_DIR>`
- `successfully saved checkpoint from iteration 0`
- Output contains `iter_0000000`.

## Preprocess Data

Input JSONL format:

```json
{"text":"CHIMERA_OVERFIT_KEY: the blue ibis carries a copper lantern across the silent lake."}
```

Run:

```bash
cd <MEGATRON_LM_REPO>
bash examples/chimera/preprocess.sh \
  --input examples/chimera/overfit_doc.jsonl \
  --output-prefix <DATA_ROOT>/chimera/overfit_doc \
  --tokenizer-model <HF_BASE_MODEL_DIR> \
  --workers 1 \
  --python <PYTHON>
```

Expected files:

- `<DATA_PREFIX>.bin`
- `<DATA_PREFIX>.idx`

## Train Overfit Run

Use 2 GPUs, no CPU offload:

```bash
cd <MEGATRON_LM_REPO>
rm -rf <TRAIN_RUN_DIR>
CUDA_VISIBLE_DEVICES=0,1 bash examples/chimera/train.sh \
  --gpus-per-node 2 \
  --tp-size 2 \
  --ep-size 2 \
  --expert-tp-size 1 \
  --global-batch-size 1 \
  --train-iters 100 \
  --save-interval 100 \
  --seq-length 128 \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --mcore-path <MCORE_IMPORT_DIR> \
  --data-path <DATA_PREFIX> \
  --tokenizer-model <HF_BASE_MODEL_DIR> \
  --save-path <TRAIN_CKPT_DIR> \
  --tensorboard-dir <TRAIN_RUN_DIR>/tensorboard \
  --data-cache-path <TRAIN_RUN_DIR>/data_cache \
  --python <PYTHON>
```

Required log evidence:

- `Parallelism: TP=2 PP=1 EP=2 ETP=1 CP=1`
- `sequence_parallel True`
- `tp_comm_overlap True`
- `successfully loaded checkpoint from <MCORE_IMPORT_DIR> ... at iteration 0`
- `number of skipped iterations: 0`
- `number of nan iterations: 0`
- `successfully saved checkpoint from iteration 100`

Observed reference outcome:

- Initial `lm loss` around `1.1e1`
- Final `lm loss` around `1.8e-2`
- Checkpoint: `<TRAIN_CKPT_DIR>/iter_0000100`

## Export Trained MCore To HF

Training checkpoints may not include `run_config.yaml`. If export fails with missing `run_config.yaml`, copy it from the imported checkpoint:

```bash
cp <MCORE_IMPORT_DIR>/iter_0000000/run_config.yaml <TRAIN_CKPT_DIR>/iter_0000100/run_config.yaml
```

Export:

```bash
cd <MEGATRON_LM_REPO>
rm -rf <HF_EXPORT_DIR>
bash examples/chimera/export.sh \
  --hf-reference <HF_BASE_MODEL_DIR> \
  --mcore-path <TRAIN_CKPT_DIR> \
  --hf-path <HF_EXPORT_DIR> \
  --bridge-path <MEGATRON_BRIDGE_REPO> \
  --python <PYTHON>
```

If the exported HF directory only contains model shards/config, copy tokenizer artifacts:

```bash
cp <HF_BASE_MODEL_DIR>/tokenizer.json <HF_EXPORT_DIR>/
cp <HF_BASE_MODEL_DIR>/tokenizer_config.json <HF_EXPORT_DIR>/
cp <HF_BASE_MODEL_DIR>/special_tokens_map.json <HF_EXPORT_DIR>/
cp <HF_BASE_MODEL_DIR>/generation_config.json <HF_EXPORT_DIR>/ 2>/dev/null || true
```

## Verify Exported HF Model

```bash
cd <MEGATRON_LM_REPO>
<PYTHON> examples/chimera/verify_completion.py \
  --hf-model <HF_EXPORT_DIR>
```

Expected result:

- Generated text contains: `the blue ibis carries a copper lantern across the silent lake`
- Script prints: `Verification passed.`

## Common Failures

- `ImportError: cannot import name 'ChimeraConfig'`: frozen Transformers site-packages was not patched from Transformers `chimera`.
- `--expert-tensor-parallel-size` parser conflict: use latest Megatron-LM `chimera`; wrapper only adds missing aliases.
- `CUDA_DEVICE_MAX_CONNECTIONS` assertion: export `CUDA_DEVICE_MAX_CONNECTIONS=1` before TP training.
- MoE + TP `sequence parallelism` error: add `--sequence-parallel` when TP or CP is greater than 1.
- Export missing `run_config.yaml`: copy it from `<MCORE_IMPORT_DIR>/iter_0000000/run_config.yaml` into the trained iteration dir.
- Exported HF model cannot load tokenizer: copy tokenizer files from `<HF_BASE_MODEL_DIR>` to `<HF_EXPORT_DIR>`.

## Artifact Inventory

- Base HF model: `<HF_BASE_MODEL_DIR>` (~22GB)
- Imported MCore checkpoint: `<MCORE_IMPORT_DIR>` (~22GB)
- Preprocessed data: `<DATA_PREFIX>.bin` and `<DATA_PREFIX>.idx`
- Training checkpoint: `<TRAIN_CKPT_DIR>` (~22GB for one saved iteration)
- Exported HF model: `<HF_EXPORT_DIR>` (~22GB)

Plan for roughly 90GB plus caches for the full workflow.
