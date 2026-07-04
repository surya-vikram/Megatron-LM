---
name: chimera-megatron-flow
description: End-to-end operational workflow for Chimera container setup, repo preparation, HF artifact creation, Megatron-LM random-init pretraining, SFT/SimPO, Megatron-Bridge conversion, HF export, and overfit verification.
---

# Chimera Megatron Flow

Use this skill when setting up Chimera in a fresh container or validating the full Transformers -> Megatron-LM -> Megatron-Bridge -> HF loop.

## Locked Architecture

- 25 decoder layers
- First 2 layers dense
- Remaining 23 layers MoE
- No final dense layer
- HF config: `first_k_dense_replace=2`, `last_k_dense_replace=0`
- Megatron pattern: `--moe-layer-freq "[0]*2+[1]*23"`
- Stable pretraining baseline: TP=1, PP=1, EP=1, ETP=1, CP=1
- 2-GPU smoke validation: TP=1, PP=1, EP=2, ETP=1, CP=1

Keep this layout consistent in:

- `transformers/src/transformers/models/chimera`
- `Megatron-Bridge` Chimera bridge code/tests
- `Megatron-LM/examples/chimera/train.sh`
- `Megatron-LM/examples/chimera/run_config.yaml`

## Fresh Container Setup

Expected container paths:

```text
/workspace/venv
/workspace/repos
```

Use persistent storage for large artifacts. On the validated H200 container, `/home/jovyan` was the persistent 477G volume and was used for checkpoints/exports:

```bash
df -h
export DATA_ROOT=/home/jovyan/chimera_smoke
```

Clone or update all three repos under `/workspace/repos`:

```bash
mkdir -p /workspace/repos

if [ ! -d /workspace/repos/Megatron-LM/.git ]; then
  git clone -b chimera https://github.com/surya-vikram/Megatron-LM.git /workspace/repos/Megatron-LM
else
  git -C /workspace/repos/Megatron-LM fetch origin chimera
  git -C /workspace/repos/Megatron-LM checkout chimera
  git -C /workspace/repos/Megatron-LM pull --ff-only origin chimera
fi

if [ ! -d /workspace/repos/Megatron-Bridge/.git ]; then
  git clone -b chimera https://github.com/surya-vikram/Megatron-Bridge.git /workspace/repos/Megatron-Bridge
else
  git -C /workspace/repos/Megatron-Bridge fetch origin chimera
  git -C /workspace/repos/Megatron-Bridge checkout chimera
  git -C /workspace/repos/Megatron-Bridge pull --ff-only origin chimera
fi

if [ ! -d /workspace/repos/transformers/.git ]; then
  git clone -b chimera https://github.com/surya-vikram/transformers.git /workspace/repos/transformers
else
  git -C /workspace/repos/transformers fetch origin chimera
  git -C /workspace/repos/transformers checkout chimera
  git -C /workspace/repos/transformers pull --ff-only origin chimera
fi
```

Point Bridge at the local Megatron-LM checkout:

```bash
ln -sfn /workspace/repos/Megatron-LM /workspace/repos/Megatron-Bridge/3rdparty/Megatron-LM
```

This may make Bridge show `M 3rdparty/Megatron-LM`; treat it as a machine-local symlink change.

Set runtime paths:

```bash
export PYTHON=/workspace/venv/bin/python3
export MEGATRON_LM=/workspace/repos/Megatron-LM
export MEGATRON_BRIDGE=/workspace/repos/Megatron-Bridge
export TRANSFORMERS=/workspace/repos/transformers
export PYTHONPATH=$MEGATRON_LM:$MEGATRON_BRIDGE/src:$TRANSFORMERS/src:${PYTHONPATH:-}
```

## Patch Installed Transformers

If the container imports installed Transformers from site-packages, patch it from the local Chimera fork:

```bash
SITE=$($PYTHON - <<'PY'
import pathlib
import transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)

cp -a /workspace/repos/transformers/src/transformers/. "$SITE/"
```

Do not rely on `rsync` in this container; the validated image did not include it.

Validate:

```bash
$PYTHON - <<'PY'
from transformers import ChimeraConfig, ChimeraForCausalLM
cfg = ChimeraConfig()
assert (cfg.first_k_dense_replace, cfg.last_k_dense_replace) == (2, 0)
print("chimera_transformers_ok", cfg.model_type, cfg.first_k_dense_replace, cfg.last_k_dense_replace)
PY
```

Optional Bridge unit test:

```bash
cd /workspace/repos/Megatron-Bridge
$PYTHON -m pytest -q tests/unit_tests/models/chimera/test_chimera_bridge.py
```

Validated result: `8 passed`.

## Create HF Artifacts

Use the real tokenizer bundled in the Transformers Chimera branch:

```bash
export HF_REFERENCE=$DATA_ROOT/hf_reference
export HF_RANDOM_FULL=$DATA_ROOT/hf_random_full
mkdir -p "$DATA_ROOT"

$PYTHON $TRANSFORMERS/src/transformers/models/chimera/scripts/export_to_hf.py \
  --output "$HF_REFERENCE" \
  --tokenizer-dir "$TRANSFORMERS/src/transformers/models/chimera/tokenizer" \
  --no-weights
```

Validate the no-weight export:

```bash
$PYTHON - <<'PY'
from transformers import AutoConfig, AutoTokenizer
import os
p = os.environ["HF_REFERENCE"]
cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(p, use_fast=True, trust_remote_code=True)
assert cfg.architectures == ["ChimeraForCausalLM"]
assert cfg.vocab_size == 50176
assert len(tok) == 50176
assert (cfg.first_k_dense_replace, cfg.last_k_dense_replace) == (2, 0)
assert tok.convert_tokens_to_ids("<start_of_turn>") == 50174
assert tok.convert_tokens_to_ids("<end_of_turn>") == 50175
assert tok.unk_token is None
assert tok.chat_template
print("hf_reference_ok", cfg.model_type, cfg.architectures, cfg.vocab_size, len(tok), bool(tok.chat_template))
PY
```

The no-weight export must include:

```text
architectures=['ChimeraForCausalLM']
vocab_size=50176
tokenizer length=50176
first_k_dense_replace=2
last_k_dense_replace=0
<start_of_turn> id=50174
<end_of_turn> id=50175
chat_template present
unk_token=None
```

Create a full random HF checkpoint only when validating HF -> MCore import:

```bash
$PYTHON $TRANSFORMERS/src/transformers/models/chimera/scripts/export_to_hf.py \
  --output "$HF_RANDOM_FULL" \
  --tokenizer-dir "$TRANSFORMERS/src/transformers/models/chimera/tokenizer" \
  --random-init \
  --dtype bfloat16 \
  --max-shard-size 5GB
```

Validated size: about `19G`.

## Import HF To MCore

```bash
export MCORE_IMPORT=$DATA_ROOT/mcore_import_config
cd "$MEGATRON_LM"
rm -rf "$MCORE_IMPORT"

bash examples/chimera/import.sh \
  --hf-model "$HF_RANDOM_FULL" \
  --mcore-path "$MCORE_IMPORT" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"
```

Expected:

```text
$MCORE_IMPORT/iter_0000000/run_config.yaml
```

## Pretraining Data Format

Pretraining JSONL is raw text only. Do not prepend BOS. `preprocess.sh` uses Megatron `tools/preprocess_data.py` with `--append-eod`, so the model sees each document followed by `<EOS>`.

Use coherent English samples for smoke validation:

```bash
export DATA_DIR=$DATA_ROOT/data
export DATA_PREFIX=$DATA_DIR/overfit_doc_text_document
mkdir -p "$DATA_DIR"

cat > "$DATA_DIR/overfit_doc.jsonl" <<'JSONL'
{"text":"CHIMERA_OVERFIT_KEY_A: The quiet engineer packed a silver notebook before sunrise and wrote down every signal from the training run."}
{"text":"CHIMERA_OVERFIT_KEY_B: A careful researcher traced the river path through the valley and marked each bridge with a blue lantern."}
JSONL
```

Preprocess:

```bash
cd "$MEGATRON_LM"
bash examples/chimera/preprocess.sh \
  --input "$DATA_DIR/overfit_doc.jsonl" \
  --output-prefix "$DATA_DIR/overfit_doc" \
  --tokenizer-model "$HF_REFERENCE" \
  --workers 8 \
  --python "$PYTHON"
```

Expected files:

```text
$DATA_PREFIX.bin
$DATA_PREFIX.idx
```

Decode the indexed dataset before training:

```bash
cd "$MEGATRON_LM"
$PYTHON - <<'PY'
import os
from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer

prefix = os.environ["DATA_PREFIX"]
tok = AutoTokenizer.from_pretrained(os.environ["HF_REFERENCE"], use_fast=True, trust_remote_code=True)
ds = indexed_dataset.IndexedDataset(prefix, multimodal=False)
print("num_sequences", len(ds))
print("document_indices", ds.document_indices.tolist())
for i in range(len(ds)):
    ids = ds[i].tolist()
    print(f"DOC_{i}_IDS", ids)
    print(f"DOC_{i}_TOKENS", tok.convert_ids_to_tokens(ids))
    print(f"DOC_{i}_DECODED", repr(tok.decode(ids, skip_special_tokens=False)))
    print(f"DOC_{i}_DECODED_SKIP_SPECIAL", repr(tok.decode(ids, skip_special_tokens=True)))
PY
```

Validated decoded documents:

```text
DOC_0_DECODED='CHIMERA_OVERFIT_KEY_A: The quiet engineer packed a silver notebook before sunrise and wrote down every signal from the training run.<EOS>'
DOC_1_DECODED='CHIMERA_OVERFIT_KEY_B: A careful researcher traced the river path through the valley and marked each bridge with a blue lantern.<EOS>'
```

## 2-GPU Smoke Overfit

The committed `train.sh` is the real pretraining script. For a 2-GPU smoke test, temporarily edit only the container copy and restore it afterward.

Apply the validated smoke edits:

```bash
cd "$MEGATRON_LM"
cp examples/chimera/train.sh /tmp/chimera_train.sh.before_smoke

$PYTHON - <<'PY'
from pathlib import Path

p = Path("examples/chimera/train.sh")
s = p.read_text()
repls = {
    "--seq-length 8192": "--seq-length 512",
    "--micro-batch-size 4": "--micro-batch-size 1",
    "--global-batch-size 16": "--global-batch-size 2",
    "--train-iters 10000": "--train-iters 200",
    "--lr 2e-4": "--lr 1e-3",
    "--min-lr 2e-5": "--min-lr 1e-4",
    "--lr-decay-iters 10000": "--lr-decay-iters 200",
    "--expert-model-parallel-size 1": "--expert-model-parallel-size 2",
    "--eval-iters 0\n": "--eval-iters 0\n    --no-save-optim\n    --no-save-rng\n",
    "TP=1 PP=1 EP=1 ETP=1 CP=1": "TP=1 PP=1 EP=2 ETP=1 CP=1",
    "seq=8192 micro=4 global=16 iters=10000": "seq=512 micro=1 global=2 iters=200",
}
for old, new in repls.items():
    if old not in s:
        raise SystemExit(f"missing expected text in train.sh: {old}")
    s = s.replace(old, new)
p.write_text(s)
PY

bash -n examples/chimera/train.sh
```

The resulting temporary diff should contain:

- `--seq-length 512`
- `--expert-model-parallel-size 2`
- `--micro-batch-size 1`
- `--global-batch-size 2`
- `--train-iters 200`
- `--lr 1e-3`
- `--min-lr 1e-4`
- `--lr-decay-iters 200`
- Add `--no-save-optim` and `--no-save-rng`
- Keep `--save-interval 1000`, `--eval-interval 1000`, `--eval-iters 0`

Run:

```bash
export RUNS_ROOT=$DATA_ROOT/runs
cd "$MEGATRON_LM"

DATA_PATH="$DATA_PREFIX" \
TOKENIZER_MODEL="$HF_REFERENCE" \
RUNS_ROOT="$RUNS_ROOT" \
bash examples/chimera/train.sh
```

Capture the run paths dynamically:

```bash
export RUN_DIR=$(ls -td "$RUNS_ROOT"/* | head -n 1)
export CHECKPOINT_DIR=$RUN_DIR/checkpoints
echo "$RUN_DIR"
```

Validated 2xH200 result:

```text
run dir: /home/jovyan/chimera_smoke/runs/20260704_114810
iteration 1 lm loss:   1.104734E+01
iteration 200 lm loss: 1.838096E-01
skipped iterations: 0
nan iterations: 0
checkpoint: checkpoints/iter_0000200
checkpoint size: about 19G
```

## Export To HF

`examples/chimera/export.sh` must install `run_config.yaml` in both:

```text
<checkpoints>/run_config.yaml
<checkpoints>/iter_<latest>/run_config.yaml
```

Bridge first checks the parent checkpoint directory but then internally switches to the latest `iter_*` directory while loading. Missing the iteration copy causes `model type None not supported`.

Run:

```bash
export HF_EXPORT=$DATA_ROOT/hf_export

cd "$MEGATRON_LM"
rm -rf "$HF_EXPORT"

bash examples/chimera/export.sh \
  --hf-reference "$HF_REFERENCE" \
  --mcore-path "$CHECKPOINT_DIR" \
  --hf-path "$HF_EXPORT" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"
```

Validated export log includes:

```text
Using Bridge run config: <checkpoints>/run_config.yaml
Using Bridge run config: <checkpoints>/iter_0000200/run_config.yaml
Successfully exported model to: <hf_export>
```

If tokenizer files are missing from the export directory, copy them from the HF reference:

```bash
for f in tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json README.md training_report.json chat_template.jinja; do
  [ -f "$HF_REFERENCE/$f" ] && cp "$HF_REFERENCE/$f" "$HF_EXPORT/$f"
done
```

Validate:

```bash
$PYTHON - <<'PY'
from transformers import AutoConfig, AutoTokenizer
import os
p = os.environ["HF_EXPORT"]
cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(p, use_fast=True, trust_remote_code=True)
assert cfg.architectures == ["ChimeraForCausalLM"]
assert (cfg.first_k_dense_replace, cfg.last_k_dense_replace) == (2, 0)
assert cfg.num_hidden_layers == 25
assert len(tok) == 50176
assert tok.convert_tokens_to_ids("<start_of_turn>") == 50174
assert tok.convert_tokens_to_ids("<end_of_turn>") == 50175
assert tok.chat_template
print("hf_export_ok", cfg.model_type, cfg.architectures, cfg.num_hidden_layers, len(tok), bool(tok.chat_template))
PY
```

## Verify Inference

```bash
cd "$MEGATRON_LM"
$PYTHON examples/chimera/verify_completion.py \
  --hf-model "$HF_EXPORT" \
  --prompt "CHIMERA_OVERFIT_KEY_A:" \
  --expected "The quiet engineer packed a silver notebook before sunrise" \
  --max-new-tokens 24
```

Validated output:

```text
CHIMERA_OVERFIT_KEY_A: The quiet engineer packed a silver notebook before sunrise and wrote down every signal from the training run.
Verification passed.
```

## Chat Tokens And Template

The Chimera tokenizer uses two non-reasoning, non-tool chat markers:

```text
<start_of_turn> id=50174
<end_of_turn>   id=50175
```

They replace low-value tail vocab entries and keep tokenizer length exactly `50176`. There is no `<unk>` token.

Chat template:

```jinja
{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content']|trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>assistant\n' }}{% endif %}
```

Validate raw rendering:

```bash
$PYTHON - <<'PY'
from transformers import AutoTokenizer
import os
tok = AutoTokenizer.from_pretrained(os.environ["HF_REFERENCE"], trust_remote_code=True)
print(tok.apply_chat_template(
    [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
    tokenize=False,
    add_generation_prompt=True,
))
PY
```

Expected:

```text
<start_of_turn>system
S<end_of_turn>
<start_of_turn>user
U<end_of_turn>
<start_of_turn>assistant
```

## SFT Smoke

SFT JSONL rows use `messages`. Do not pass `--pack-samples`; the Chimera prompt format masks system, user, and assistant header tokens, and trains only assistant content plus `<end_of_turn>`.

Create two samples:

```bash
export CHAT_ROOT=/home/jovyan/chimera_chat_smoke
export SFT_DATA=$CHAT_ROOT/data/sft.jsonl
mkdir -p "$CHAT_ROOT/data"

cat > "$SFT_DATA" <<'JSONL'
{"messages":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SFT key A response?"},{"role":"assistant","content":"CHIMERA_SFT_RESPONSE_A: jade lanterns align under quiet stars."}]}
{"messages":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SFT key B response?"},{"role":"assistant","content":"CHIMERA_SFT_RESPONSE_B: silver rivers circle patient mountains."}]}
JSONL
```

Run from an MCore checkpoint:

```bash
cd "$MEGATRON_LM"
DATA_PATH="$SFT_DATA" \
TOKENIZER_MODEL="$HF_REFERENCE" \
MCORE_PATH="$CHECKPOINT_DIR" \
RUNS_ROOT="$CHAT_ROOT/sft_runs" \
SEQ_LENGTH=128 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
TRAIN_ITERS=120 \
LR=1e-3 \
MIN_LR=1e-4 \
LR_WARMUP_ITERS=0 \
LR_DECAY_ITERS=120 \
SAVE_INTERVAL=1000 \
EVAL_INTERVAL=1000 \
EVAL_ITERS=0 \
SAVE_WEIGHTS_ONLY=true \
bash examples/chimera/sft.sh
```

Validated result:

```text
sft_tokenizer_prompt_format chimera
pack_samples False
iteration 1 lm loss:   1.770898E+01
iteration 120 lm loss: 6.497540E-04
checkpoint: <sft_run>/checkpoints/iter_0000120
```

Export the SFT checkpoint exactly like pretraining export, then copy tokenizer artifacts including `chat_template.jinja` from `HF_REFERENCE`.

Raw inference should keep special tokens visible:

```text
RAW_SFT_A= <start_of_turn>system
You answer with the exact requested phrase.<end_of_turn>
<start_of_turn>user
What is the Chimera SFT key A response?<end_of_turn>
<start_of_turn>assistant
CHIMERA_SFT_RESPONSE_A: jade lanterns align under quiet stars.<end_of_turn>
```

## SimPO Smoke

SimPO JSONL rows use `chosen` and `rejected`, each as a messages list. Do not pass `--pack-samples`.

For tiny overfit smoke, the generic blended dataset builder must have at least as many rows as requested training samples. Keep two unique examples but repeat them in the JSONL when using many iterations.

Create a repeated two-example dataset:

```bash
export SIMPO_DATA=$CHAT_ROOT/data/simpo_repeated.jsonl
$PYTHON - <<'PY'
import json
import os
from pathlib import Path

examples = [
    {"chosen":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key C response?"},{"role":"assistant","content":"CHIMERA_SIMPO_CHOSEN_C: amber maps reward careful answers."}],"rejected":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key C response?"},{"role":"assistant","content":"This is the rejected answer for C."}]},
    {"chosen":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key D response?"},{"role":"assistant","content":"CHIMERA_SIMPO_CHOSEN_D: violet signals favor steady choices."}],"rejected":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key D response?"},{"role":"assistant","content":"This is the rejected answer for D."}]},
]
path = Path(os.environ["SIMPO_DATA"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w") as f:
    for i in range(120):
        f.write(json.dumps(examples[i % 2]) + "\n")
PY
```

Run from the SFT MCore checkpoint:

```bash
cd "$MEGATRON_LM"
DATA_PATH="$SIMPO_DATA" \
TOKENIZER_MODEL="$HF_REFERENCE" \
MCORE_PATH="$SFT_CHECKPOINT_DIR" \
RUNS_ROOT="$CHAT_ROOT/simpo_runs" \
SEQ_LENGTH=128 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
TRAIN_ITERS=40 \
LR=5e-4 \
MIN_LR=5e-5 \
LR_WARMUP_ITERS=0 \
LR_DECAY_ITERS=40 \
SAVE_INTERVAL=1000 \
EVAL_INTERVAL=1000 \
EVAL_ITERS=0 \
SAVE_WEIGHTS_ONLY=true \
SIMPO_SFT_WEIGHT=1.0 \
bash examples/chimera/simpo.sh
```

Validated result:

```text
pack_samples False
iteration 1 simpo loss:  1.970660E+01
iteration 40 simpo loss: 4.444013E-02
iteration 40 rewards/accuracies: 1.000000E+00
checkpoint: <simpo_run>/checkpoints/iter_0000040
```

Export the SimPO checkpoint and copy tokenizer artifacts including `chat_template.jinja`.

Raw inference should keep special tokens visible:

```text
RAW_SIMPO_C= <start_of_turn>system
You answer with the exact requested phrase.<end_of_turn>
<start_of_turn>user
What is the Chimera SimPO key C response?<end_of_turn>
<start_of_turn>assistant
CHIMERA_SIMPO_CHOSEN_C: amber maps reward careful answers.<end_of_turn>
```

Restore the committed training script after the smoke run:

```bash
cd "$MEGATRON_LM"
if [ -f /tmp/chimera_train.sh.before_smoke ]; then
  cp /tmp/chimera_train.sh.before_smoke examples/chimera/train.sh
else
  git restore examples/chimera/train.sh
fi
git status --short
```

## Hard Rules

- Do not add BOS to pretraining JSONL.
- Pretraining document boundaries come from `--append-eod`.
- Keep real pretraining checkpoints resumable; use `--no-save-optim` and `--no-save-rng` only for short smoke runs.
- Do not write large artifacts to root overlay if persistent storage exists.
- Restore temporary `train.sh` edits after smoke validation.
- If export fails with missing architecture, regenerate HF reference from a Transformers commit where no-weight export writes `architectures=["ChimeraForCausalLM"]`.
- If export fails with `model type None not supported`, ensure `run_config.yaml` exists in both checkpoint root and latest `iter_*`.
