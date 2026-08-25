# Chimera End-to-End Runbook

The single-container commands below reproduce the complete validation flow.
For production, `cluster_manager.sh` launches the same Chimera scripts on one
to N nodes with a shared run identifier and persistent storage.

## 0. Cluster Container Launch

The validated base image is:

```text
suryavikram6/megatron-gemma:v2-fixed
```

Prepare this shared host layout at the same path on every selected node:

```text
/nvme_zone3/home/ekamai1/surya/chimera/
├── repos/
│   ├── Megatron-LM/
│   ├── Megatron-Bridge/
│   └── transformers/
└── data/
    ├── hf_models/
    ├── pretrain/
    ├── sft/
    ├── simpo/
    ├── checkpoints/
    ├── runs/
    ├── exports/
    ├── logs/
    └── cache/huggingface/
```

The manager mounts the two roots as:

```text
.../chimera/repos -> /workspace/repos
.../chimera/data  -> /datasets/megadata
```

Point Bridge at the mounted Megatron-LM checkout before launch:

```bash
ln -sfn ../../Megatron-LM \
  /nvme_zone3/home/ekamai1/surya/chimera/repos/Megatron-Bridge/3rdparty/Megatron-LM
```

Show the complete command guide:

```bash
bash examples/chimera/cluster_manager.sh --help
```

Create one configuration per run outside the Git checkout:

```bash
cp examples/chimera/cluster.env.example \
  /nvme_zone3/home/ekamai1/surya/chimera/data/pretrain-phase1.env
```

Edit the stage, run name, node list, input paths, checkpoint path, topology,
and batch sizes. Then inspect and validate before launching:

```bash
bash examples/chimera/cluster_manager.sh \
  --config /nvme_zone3/home/ekamai1/surya/chimera/data/pretrain-phase1.env info

bash examples/chimera/cluster_manager.sh \
  --config /nvme_zone3/home/ekamai1/surya/chimera/data/pretrain-phase1.env preflight

bash examples/chimera/cluster_manager.sh \
  --config /nvme_zone3/home/ekamai1/surya/chimera/data/pretrain-phase1.env launch
```

Monitor or stop the same run by reusing the same configuration:

```bash
bash examples/chimera/cluster_manager.sh --config /path/to/run.env status
bash examples/chimera/cluster_manager.sh --config /path/to/run.env logs
bash examples/chimera/cluster_manager.sh --config /path/to/run.env stop
bash examples/chimera/cluster_manager.sh --config /path/to/run.env cleanup
```

The manager does not clone, pull, or modify repositories. It does not source
the image's malformed `/workspace/load_env.sh`; it explicitly selects
`/workspace/venv` and the three mounted source trees. SFT always starts from a
pretraining MCore checkpoint, and SimPO always starts from an SFT MCore
checkpoint. Only pretraining retains checkpoint-resume behavior.

For one node, set `NODES_CSV` to one hostname. For additional nodes, append
comma-separated hostnames; list order determines `NODE_RANK`, and the first
node is the master. All selected nodes must have the same GPU count. On a
CPU-only host, use `image-check`, `preflight`, and `dry-run`; `launch` requires
GPUs.

## 1. Environment

```bash
export PYTHON=/workspace/venv/bin/python3
export REPOS=/workspace/repos
export MEGATRON_LM=$REPOS/Megatron-LM
export MEGATRON_BRIDGE=$REPOS/Megatron-Bridge
export TRANSFORMERS=$REPOS/transformers
export DATA_ROOT=/datasets/megadata/chimera_validation
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
assert cfg.rms_norm_eps == 1e-5
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
assert (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (32, 4, 2048)
assert cfg.n_shared_experts == 0 and cfg.shared_expert_intermediate_size == 0
assert cfg.qk_layernorm and cfg.max_position_embeddings == 8192
assert cfg.rope_parameters["factor"] == 1.0
assert cfg.router_load_balancing_type == "quantile_balancing"
assert (cfg.moe_qb_num_bins, cfg.moe_qb_ema_decay) == (1000, 0.0)
assert cfg.load_with_bias is True
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
For both conversion cycles and exact key/shape/dtype/value/SHA256 reports, use
`examples/chimera/verify_conversion.sh` as documented in the workflow skill.

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

### Optimized preprocessing (opt-in)

The commands above remain the default and run `tools/preprocess_data.py`. For
large datasets, add `--optimized` to run dedicated reader, tokenizer, and
shard-writer process pools. The optimized path splits JSONL files into
newline-safe byte ranges and parquet files into row-group tasks, connects the
stages with byte-bounded queues, and merges independent writer shards into one
final `.bin/.idx` pair:

```bash
export PREPROCESS_READERS=10
export PREPROCESS_TOKENIZERS=192
export PREPROCESS_WRITERS=6
export PREPROCESS_QUEUE_MEMORY_BUDGET_GB=280

cd "$MEGATRON_LM"
bash examples/chimera/preprocess.sh \
  --optimized \
  --input "$PRETRAIN_SOURCE_DIR" \
  --output-prefix "$PRETRAIN_OUTPUT_DIR/fineweb_edu" \
  --tokenizer-model "$HF_REFERENCE" \
  --num-readers "$PREPROCESS_READERS" \
  --num-tokenizers "$PREPROCESS_TOKENIZERS" \
  --num-writers "$PREPROCESS_WRITERS" \
  --queue-memory-budget-gb "$PREPROCESS_QUEUE_MEMORY_BUDGET_GB" \
  --log-interval-seconds 30 \
  --python "$PYTHON"
```

The three optimized process counts are independent; there is no separate total
for this path. On the 240-logical-CPU reference host, `10 + 192 + 6 = 208`
leaves 32 logical CPUs for the coordinator, runtime threads, filesystem work,
and the operating system. The queue-memory budget is divided equally between
raw and encoded queue byte credits, so `280` permits up to 140 GiB of estimated
payload in each queue. It does not include tokenizer-process or PyArrow memory.
Optimized progress logs show documents read, tokenized, and written; tokens
written; both queue depths; and token throughput. The final `PIPELINE_SUMMARY`
separates processing, shard-merge, and total time.

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

`train.sh` is the production 8k script. It accepts schedule overrides through
the environment, so the validated 2-GPU smoke does not edit tracked files.

Use 800 iterations at sequence length 128 for this full-model two-document
check. A 400-iteration run at sequence length 512 reached low teacher-forced
loss but did not reproduce the complete B continuation from its bare prefix.

Run random-init pretraining:

```bash
export PRETRAIN_RUNS=$DATA_ROOT/pretrain_runs
TRAIN_DATA_PATH="$DATA_PREFIX" \
TOKENIZER_MODEL="$HF_REFERENCE" \
RUNS_ROOT="$PRETRAIN_RUNS" \
INTRA_DOC_MASKING=true \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
SEQ_LENGTH=128 \
TRAIN_ITERS=800 \
LR=1e-3 \
MIN_LR=1e-4 \
LR_DECAY_STYLE=cosine \
LR_WARMUP_ITERS=0 \
WEIGHT_DECAY=0.0 \
SAVE_INTERVAL=800 \
TP_SIZE=1 PP_SIZE=1 EP_SIZE=1 CP_SIZE=1 \
bash examples/chimera/train.sh

export PRETRAIN_RUN_DIR=$(ls -td "$PRETRAIN_RUNS"/* | head -n 1)
export PRETRAIN_CHECKPOINT=$PRETRAIN_RUN_DIR/checkpoints
```

The production default remains `INTRA_DOC_MASKING=false`. This two-document
memorization test enables it so the bare A and B prefixes are each trained as
independent contexts; otherwise B is only observed after A in the packed
sequence and standalone B generation is not a valid memorization check.

With two visible GPUs this is DP=2. On a measured OOM, first add
`MAIN_GRADS_DTYPE=bf16 EXP_AVG_DTYPE=bf16 EXP_AVG_SQ_DTYPE=bf16`. If that still
fails, use exactly one fallback axis, `EP_SIZE=2` or `TP_SIZE=2`, keeping the
other at 1. Never use EP=2 and TP=2 together on two GPUs. Since the smoke uses
environment-only overrides, `git diff --exit-code -- examples/chimera/train.sh`
must remain clean afterward.

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

For stateless interactive Transformers inference, load the model once and
enter independent prompts. Both commands use the same checkpoint and frozen
bias tensors; only their use during expert selection changes in memory:

```bash
$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_PRETRAIN" --interactive --chat --stream \
  --load-with-bias true --dtype bfloat16 --device-map auto

$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_PRETRAIN" --interactive --chat --stream \
  --load-with-bias false --dtype bfloat16 --device-map auto
```

Use `/exit`, `/quit`, Ctrl-C, or Ctrl-D to stop. An optional fixed
`--system-prompt` is reapplied independently to every prompt; earlier user and
assistant turns are never retained. Generation is greedy by default. Add, for
example, `--do-sample --temperature 0.7 --top-p 0.9 --top-k 50
--repetition-penalty 1.1` to enable sampling.

## 7. SFT

SFT reads JSONL directly. Each checked fixture row contains exactly one
conversation:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The loss mask trains assistant content and `<end_of_turn>`, while masking the
system turn, user turn, and assistant header. Production runs pack samples by
default; this exact-response smoke disables packing explicitly.

```bash
export SFT_RUNS=$DATA_ROOT/sft_runs
DATA_PATH="$MEGATRON_LM/examples/chimera/data/sft/overfit.jsonl" \
TOKENIZER_MODEL="$HF_REFERENCE" \
MCORE_PATH="$PRETRAIN_CHECKPOINT" \
RUNS_ROOT="$SFT_RUNS" \
PACK_SAMPLES=false \
SEQ_LENGTH=128 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
TRAIN_ITERS=120 \
LR=1e-3 MIN_LR=1e-4 LR_WARMUP_ITERS=0 LR_DECAY_ITERS=120 \
SAVE_INTERVAL=120 EVAL_INTERVAL=1000 EVAL_ITERS=0 \
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

For custom Transformers inference, request a dictionary and pass its tensors
to `generate()` explicitly. Do not pass the complete `BatchEncoding` as the
positional `inputs` argument:

```python
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.environ["HF_SFT"]
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You answer with the exact requested phrase."},
    {"role": "user", "content": "What is the Chimera SFT key A response?"},
]
encoded = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
)
input_ids = encoded["input_ids"].to(model.device)
attention_mask = encoded["attention_mask"].to(model.device)
output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=48,
    do_sample=False,
    eos_token_id=[
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    ],
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False))
```

## 8. SimPO

SimPO also reads JSONL directly and does not use `preprocess.sh`:

```json
{"chosen":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"preferred"}],"rejected":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"rejected"}]}
```

The checked file contains two unique C/D preference pairs plus two explicit
A/B retention-replay pairs. The replay rows keep the accepted SFT responses as
their chosen answers; `SIMPO_SFT_WEIGHT` alone is not a retention mechanism,
because it applies chosen-answer loss only to rows in the SimPO dataset.
`SimPODataset` reports both the physical row count and requested logical sample
count and wraps over the four physical rows until the requested training length
is satisfied. Do not duplicate the JSONL for a smoke run:

```bash
export SIMPO_DATA=$MEGATRON_LM/examples/chimera/data/simpo/overfit.jsonl
test "$(wc -l < "$SIMPO_DATA")" -eq 4
```

Run SimPO from the SFT checkpoint. Production runs pack samples by default;
this retention smoke disables packing explicitly:

```bash
export SIMPO_RUNS=$DATA_ROOT/simpo_runs
DATA_PATH="$SIMPO_DATA" \
TOKENIZER_MODEL="$HF_REFERENCE" \
MCORE_PATH="$SFT_CHECKPOINT" \
RUNS_ROOT="$SIMPO_RUNS" \
PACK_SAMPLES=false \
SEQ_LENGTH=128 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
TRAIN_ITERS=80 \
LR=5e-4 MIN_LR=5e-5 LR_WARMUP_ITERS=0 LR_DECAY_ITERS=80 \
SAVE_INTERVAL=80 EVAL_INTERVAL=1000 EVAL_ITERS=0 \
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

Also rerun the SFT A and B prompts from section 7. Both accepted SFT responses
must remain exact after SimPO; a C/D-only fixture can pass preference accuracy
while catastrophically forgetting them.

## 9. Completion Criteria

- The HF reference has the finalized 50176-token tokenizer and chat template.
- HF-to-MCore conversion writes `iter_0000000/run_config.yaml`.
- Decoded pretraining documents end in `<EOS>` and contain no inserted BOS.
- Random-init pretraining loss converges and the exported HF model memorizes A and B through `<EOS>`.
- SFT raw inference emits the exact A and B answers followed by `<end_of_turn>`.
- SimPO reward accuracy reaches 1.0 and raw inference emits the chosen C and D answers.
- `examples/chimera/train.sh` remains unchanged; smoke overrides are recorded in `run_paths.env`.
