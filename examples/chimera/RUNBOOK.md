# Chimera End-to-End Runbook

Run all commands from a container with two H200 GPUs. The repositories are
expected at `/workspace/repos` on their `chimera` branches. Store large
artifacts on a persistent volume, not under a Git checkout.

## 1. Environment

```bash
export PYTHON=/workspace/venv/bin/python3
export REPOS=/workspace/repos
export MEGATRON_LM=$REPOS/Megatron-LM
export MEGATRON_BRIDGE=$REPOS/Megatron-Bridge
export TRANSFORMERS=$REPOS/transformers
export DATA_ROOT=/home/jovyan/chimera_validation
export PYTHONPATH=$MEGATRON_LM:$MEGATRON_BRIDGE/src:$TRANSFORMERS/src:${PYTHONPATH:-}

mkdir -p "$DATA_ROOT"
ln -sfn "$MEGATRON_LM" "$MEGATRON_BRIDGE/3rdparty/Megatron-LM"
```

Confirm that Python is loading the Chimera fork:

```bash
$PYTHON - <<'PY'
from transformers import ChimeraConfig, ChimeraForCausalLM
cfg = ChimeraConfig()
assert (cfg.first_k_dense_replace, cfg.last_k_dense_replace) == (2, 0)
print("transformers_ok", cfg.model_type, cfg.first_k_dense_replace, cfg.last_k_dense_replace)
PY
```

If the container forces the installed Transformers package ahead of
`PYTHONPATH`, replace it once from the fork. The guard avoids copying the source
tree onto itself when the fork is already being imported:

```bash
SITE=$($PYTHON - <<'PY'
import pathlib, transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)
SOURCE=$(realpath "$TRANSFORMERS/src/transformers")
if [ "$(realpath "$SITE")" != "$SOURCE" ]; then
  cp -a "$SOURCE/." "$SITE/"
fi
```

## 2. Create HF Artifacts

The Transformers branch contains the finalized tokenizer, chat template, model
configuration, generation configuration, and inference script.

```bash
export HF_REFERENCE=$DATA_ROOT/hf_reference
export HF_RANDOM_FULL=$DATA_ROOT/hf_random_full

rm -rf "$HF_REFERENCE"
$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/export_to_hf.py" \
  --output "$HF_REFERENCE" \
  --no-weights
```

The bundled tokenizer is used automatically. Validate it:

```bash
$PYTHON - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer

p = os.environ["HF_REFERENCE"]
cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(p, use_fast=True, trust_remote_code=True)
assert cfg.architectures == ["ChimeraForCausalLM"]
assert (cfg.first_k_dense_replace, cfg.last_k_dense_replace) == (2, 0)
assert cfg.vocab_size == len(tok) == 50176
assert tok.convert_tokens_to_ids("<start_of_turn>") == 2
assert tok.convert_tokens_to_ids("<end_of_turn>") == 3
assert tok.unk_token is None and tok.chat_template
print("hf_reference_ok", cfg.model_type, len(tok), bool(tok.chat_template))
PY
```

Create full random HF weights only to validate HF-to-MCore conversion:

```bash
rm -rf "$HF_RANDOM_FULL"
$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/export_to_hf.py" \
  --output "$HF_RANDOM_FULL" \
  --random-init \
  --dtype bfloat16 \
  --max-shard-size 5GB
```

## 3. Convert HF To MCore

```bash
export MCORE_IMPORT=$DATA_ROOT/mcore_import
cd "$MEGATRON_LM"
rm -rf "$MCORE_IMPORT"

bash examples/chimera/import.sh \
  --hf-model "$HF_RANDOM_FULL" \
  --mcore-path "$MCORE_IMPORT" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"

test -f "$MCORE_IMPORT/iter_0000000/run_config.yaml"
```

This import validates the conversion path. Pretraining below intentionally
starts from Megatron random initialization and does not load this checkpoint.

## 4. Pretraining Data

The checked source schema is one raw document per JSONL row:

```json
{"text":"CHIMERA_OVERFIT_KEY_A: The quiet engineer ..."}
{"text":"CHIMERA_OVERFIT_KEY_B: A careful researcher ..."}
```

For production data, `--input` may also be a parquet file, glob, or directory.
Directories are traversed recursively for `.parquet` and `.jsonl` files. The
script reports each format count, processes paths in deterministic sorted
order, and writes all documents into one `.bin/.idx` pair. Parquet input is
streamed in record batches and must contain the configured `text` column.
`examples/chimera/preprocess.sh` is a Chimera-specific wrapper around the
generic `tools/preprocess_data.py`: it selects `HuggingFaceTokenizer`, passes
the Chimera tokenizer path, and always appends one `<EOS>` document boundary.
It does not format chat data or run SFT/SimPO.

Regenerate the committed `.bin/.idx` files whenever the JSONL or tokenizer
changes:

```bash
cd "$MEGATRON_LM"
bash examples/chimera/preprocess.sh \
  --tokenizer-model "$HF_REFERENCE" \
  --workers 8 \
  --python "$PYTHON"
```

For a production directory containing parquet and JSONL files at any depth,
use one output prefix. This command recursively discovers both formats and
combines every document into one indexed dataset:

```bash
export PRETRAIN_SOURCE_DIR=/datasets/fineweb-edu
export PRETRAIN_OUTPUT_DIR=/datasets/processed/fineweb-edu
export TOTAL_CPUS=$(nproc --all)
export PREPROCESS_WORKERS=$((TOTAL_CPUS > 8 ? TOTAL_CPUS - 8 : 1))
mkdir -p "$PRETRAIN_OUTPUT_DIR"
"$PYTHON" -c "import pyarrow"

cd "$MEGATRON_LM"
bash examples/chimera/preprocess.sh \
  --input "$PRETRAIN_SOURCE_DIR" \
  --output-prefix "$PRETRAIN_OUTPUT_DIR/fineweb_edu" \
  --tokenizer-model "$HF_REFERENCE" \
  --workers "$PREPROCESS_WORKERS" \
  --log-interval 10000 \
  --log-interval-seconds 30 \
  --parquet-batch-size 1024 \
  --python "$PYTHON"

export DATA_PREFIX=$PRETRAIN_OUTPUT_DIR/fineweb_edu_text_document
test -f "$DATA_PREFIX.bin"
test -f "$DATA_PREFIX.idx"
```

The discovery log must report the expected parquet and JSONL file counts
and exact document count before tokenization starts. Progress logs include
percentage, throughput, elapsed time, and ETA every 30 seconds. Only files
ending in `.parquet` or `.jsonl` are included. Pass `--skip-document-count` to
avoid the initial JSONL line scan; percentage and ETA are then unavailable.
ETA is a smoothed estimate based on documents per second, so it will move when
document lengths or tokenizer cost vary across shards.

Inspect exactly what Megatron reads. Each decoded document must end in `<EOS>`;
there must be no inserted `<BOS>`:

```bash
export DATA_PREFIX=$MEGATRON_LM/examples/chimera/data/pretrain/overfit_text_document
$PYTHON - <<'PY'
import os
from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer

ds = indexed_dataset.IndexedDataset(os.environ["DATA_PREFIX"], multimodal=False)
tok = AutoTokenizer.from_pretrained(os.environ["HF_REFERENCE"], use_fast=True)
print("num_documents", len(ds), "document_indices", ds.document_indices.tolist())
for i in range(len(ds)):
    ids = ds[i].tolist()
    print(f"DOC_{i}_IDS", ids)
    print(f"DOC_{i}_DECODED", repr(tok.decode(ids, skip_special_tokens=False)))
PY
```

SFT and SimPO do not use this preprocessing step.

## 5. Pretraining Overfit

`train.sh` is the production 8k script. On the disposable container checkout,
temporarily change it to the validated 2-GPU smoke configuration:

Use 400 iterations for this two-document check. At 200 iterations the
teacher-forced loss was low, but the bare A and B prefixes still tied on their
first continuation token and did not both generate correctly.

```bash
cd "$MEGATRON_LM"
cp examples/chimera/train.sh /tmp/chimera_train.sh.before_smoke

$PYTHON - <<'PY'
from pathlib import Path

p = Path("examples/chimera/train.sh")
s = p.read_text()
replacements = {
    "--seq-length 8192": "--seq-length 512",
    "--micro-batch-size 4": "--micro-batch-size 1",
    "--global-batch-size 16": "--global-batch-size 2",
    "--train-iters 10000": "--train-iters 400",
    "--lr 2e-4": "--lr 1e-3",
    "--min-lr 2e-5": "--min-lr 1e-4",
    "--lr-decay-iters 10000": "--lr-decay-iters 400",
    "--expert-model-parallel-size 1": "--expert-model-parallel-size 2",
    "--eval-iters 0\n": "--eval-iters 0\n    --no-save-optim\n    --no-save-rng\n",
    "TP=1 PP=1 EP=1 ETP=1 CP=1": "TP=1 PP=1 EP=2 ETP=1 CP=1",
    "seq=8192 micro=4 global=16 iters=10000": "seq=512 micro=1 global=2 iters=400",
}
for old, new in replacements.items():
    if old not in s:
        raise SystemExit(f"Expected train.sh text is missing: {old}")
    s = s.replace(old, new)
p.write_text(s)
PY

bash -n examples/chimera/train.sh
```

Run random-init pretraining:

```bash
export PRETRAIN_RUNS=$DATA_ROOT/pretrain_runs
DATA_PATH="$DATA_PREFIX" \
TOKENIZER_MODEL="$HF_REFERENCE" \
RUNS_ROOT="$PRETRAIN_RUNS" \
bash examples/chimera/train.sh

export PRETRAIN_RUN_DIR=$(ls -td "$PRETRAIN_RUNS"/* | head -n 1)
export PRETRAIN_CHECKPOINT=$PRETRAIN_RUN_DIR/checkpoints
```

Restore the production script immediately after the smoke run:

```bash
cp /tmp/chimera_train.sh.before_smoke examples/chimera/train.sh
git diff --exit-code -- examples/chimera/train.sh
```

## 6. Export And Verify Pretraining

```bash
export HF_PRETRAIN=$DATA_ROOT/hf_pretrain
rm -rf "$HF_PRETRAIN"

bash examples/chimera/export.sh \
  --hf-reference "$HF_REFERENCE" \
  --mcore-path "$PRETRAIN_CHECKPOINT" \
  --hf-path "$HF_PRETRAIN" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"

$PYTHON examples/chimera/verify_pretrain.py \
  --hf-model "$HF_PRETRAIN" \
  --prompt "CHIMERA_OVERFIT_KEY_A:" \
  --expected " The quiet engineer packed a silver notebook before sunrise and wrote down every signal from the training run.<EOS>"

$PYTHON examples/chimera/verify_pretrain.py \
  --hf-model "$HF_PRETRAIN" \
  --prompt "CHIMERA_OVERFIT_KEY_B:" \
  --expected " A careful researcher traced the river path through the valley and marked each bridge with a blue lantern.<EOS>"
```

## 7. SFT

SFT reads JSONL directly. Each checked fixture row contains exactly one
conversation:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The loss mask trains assistant content and `<end_of_turn>`, while masking the
system turn, user turn, and assistant header. Packing is disabled.

```bash
export SFT_RUNS=$DATA_ROOT/sft_runs
DATA_PATH="$MEGATRON_LM/examples/chimera/data/sft/overfit.jsonl" \
TOKENIZER_MODEL="$HF_REFERENCE" \
MCORE_PATH="$PRETRAIN_CHECKPOINT" \
RUNS_ROOT="$SFT_RUNS" \
SEQ_LENGTH=128 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
TRAIN_ITERS=120 \
LR=1e-3 MIN_LR=1e-4 LR_WARMUP_ITERS=0 LR_DECAY_ITERS=120 \
SAVE_INTERVAL=1000 EVAL_INTERVAL=1000 EVAL_ITERS=0 \
SAVE_WEIGHTS_ONLY=true \
bash examples/chimera/sft.sh

export SFT_RUN_DIR=$(ls -td "$SFT_RUNS"/* | head -n 1)
export SFT_CHECKPOINT=$SFT_RUN_DIR/checkpoints
export HF_SFT=$DATA_ROOT/hf_sft
rm -rf "$HF_SFT"

bash examples/chimera/export.sh \
  --hf-reference "$HF_REFERENCE" \
  --mcore-path "$SFT_CHECKPOINT" \
  --hf-path "$HF_SFT" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"
```

Run the shared HF `infer.py`. `--show-special-tokens` makes the terminating
`<end_of_turn>` visible in the raw output:

```bash
$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SFT" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SFT key A response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto

$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SFT" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SFT key B response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto
```

Expected completion:

```text
CHIMERA_SFT_RESPONSE_A: jade lanterns align under quiet stars.<end_of_turn>
CHIMERA_SFT_RESPONSE_B: silver rivers circle patient mountains.<end_of_turn>
```

## 8. SimPO

SimPO also reads JSONL directly and does not use `preprocess.sh`:

```json
{"chosen":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"preferred"}],"rejected":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"rejected"}]}
```

The checked file contains two unique preference pairs. Repeat them into the
working data file because the generic blended-dataset builder requires at least
as many rows as requested training samples:

```bash
export SIMPO_DATA=$DATA_ROOT/simpo_overfit_repeated.jsonl
$PYTHON - <<'PY'
import json, os
from pathlib import Path

source = Path(os.environ["MEGATRON_LM"]) / "examples/chimera/data/simpo/overfit.jsonl"
examples = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
assert len(examples) == 2
with open(os.environ["SIMPO_DATA"], "w") as out:
    for i in range(120):
        out.write(json.dumps(examples[i % 2]) + "\n")
PY
```

Run SimPO from the SFT checkpoint. Packing is disabled:

```bash
export SIMPO_RUNS=$DATA_ROOT/simpo_runs
DATA_PATH="$SIMPO_DATA" \
TOKENIZER_MODEL="$HF_REFERENCE" \
MCORE_PATH="$SFT_CHECKPOINT" \
RUNS_ROOT="$SIMPO_RUNS" \
SEQ_LENGTH=128 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
TRAIN_ITERS=40 \
LR=5e-4 MIN_LR=5e-5 LR_WARMUP_ITERS=0 LR_DECAY_ITERS=40 \
SAVE_INTERVAL=1000 EVAL_INTERVAL=1000 EVAL_ITERS=0 \
SAVE_WEIGHTS_ONLY=true \
SIMPO_SFT_WEIGHT=1.0 \
bash examples/chimera/simpo.sh

export SIMPO_RUN_DIR=$(ls -td "$SIMPO_RUNS"/* | head -n 1)
export SIMPO_CHECKPOINT=$SIMPO_RUN_DIR/checkpoints
export HF_SIMPO=$DATA_ROOT/hf_simpo
rm -rf "$HF_SIMPO"

bash examples/chimera/export.sh \
  --hf-reference "$HF_REFERENCE" \
  --mcore-path "$SIMPO_CHECKPOINT" \
  --hf-path "$HF_SIMPO" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"
```

Verify the preferred response with the same `infer.py`:

```bash
$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SIMPO" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SimPO key C response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto

$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SIMPO" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SimPO key D response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto
```

Expected completion:

```text
CHIMERA_SIMPO_CHOSEN_C: amber maps reward careful answers.<end_of_turn>
CHIMERA_SIMPO_CHOSEN_D: violet signals favor steady choices.<end_of_turn>
```

## 9. Completion Criteria

- The HF reference has the finalized 50176-token tokenizer and chat template.
- HF-to-MCore conversion writes `iter_0000000/run_config.yaml`.
- Decoded pretraining documents end in `<EOS>` and contain no inserted BOS.
- Random-init pretraining loss converges and the exported HF model memorizes A and B through `<EOS>`.
- SFT raw inference emits the exact A and B answers followed by `<end_of_turn>`.
- SimPO reward accuracy reaches 1.0 and raw inference emits the chosen C and D answers.
- `examples/chimera/train.sh` is restored to the committed 8k configuration.
