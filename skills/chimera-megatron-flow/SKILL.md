---
name: chimera-megatron-flow
description: End-to-end operational workflow for Chimera container setup, repo preparation, HF artifact creation, Megatron-LM random-init pretraining, SFT/SimPO, Megatron-Bridge conversion, HF export, and overfit verification.
---

# Chimera Megatron Flow

Use this skill when setting up Chimera in a fresh container or validating the full Transformers -> Megatron-LM -> Megatron-Bridge -> HF loop.

For the checked-in sample data and concise copy-paste commands, also read
`examples/chimera/RUNBOOK.md`. Keep that runbook and this detailed recovery
guide consistent.

## Locked Architecture

- 25 decoder layers
- First 2 layers dense
- Remaining 23 layers MoE
- No final dense layer
- HF config: `first_k_dense_replace=2`, `last_k_dense_replace=0`
- Megatron pattern: `--moe-layer-freq "[0]*2+[1]*23"`
- Hidden size 2048, dense FFN size 8192
- 16 attention heads, 2 query groups, head dimension 256, QK RMSNorm enabled
- All RMSNorm sites use epsilon `1e-5`
- 32 routed experts, top-4, expert FFN size 2048, no shared expert
- Sigmoid routing with scaling factor 2.5
- Pretraining uses quantile balancing, 1000 bins, EMA 0.0, aux 0.0, and z-loss 0.001
- Every checkpoint contains frozen `e_score_correction_bias` tensors; `load_with_bias` controls use, not loading
- SFT and SimPO use load balancing `none` and bias update rate 0.0
- Maximum/original context 8192, YaRN factor 1.0, `mscale=1.0`, `mscale_all_dim=0.0`
- Stable pretraining baseline: TP=1, PP=1, EP=1, ETP=1, CP=1
- 2-GPU validation starts as DP=2 with TP=1, PP=1, EP=1, ETP=1, CP=1
- On measured OOM, first reduce optimizer moments to BF16; then use EP=2 or TP=2 as one fallback axis, never both

The reduced canonical tiny profile is 8 layers (`[0]*2+[1]*6`), hidden 512,
dense FFN 2048, 8 heads, 2 query groups, head dimension 64, 8 routed experts,
top-2, expert FFN 256, and the same QK norm, no-shared-expert, bias, QB, and
8K/factor-1 behavior.

Keep this layout consistent in:

- `transformers/src/transformers/models/chimera`
- `Megatron-Bridge` Chimera bridge code/tests
- `Megatron-LM/examples/chimera/train.sh`
- `Megatron-LM/examples/chimera/run_config.yaml`

## Cluster Container Manager

The validated base image is `suryavikram6/megatron-gemma:v2-fixed`. For
one-to-N node training, use `examples/chimera/cluster_manager.sh` from a
controller with passwordless SSH to every selected node. Show its command
guide with:

```bash
bash examples/chimera/cluster_manager.sh --help
```

The shared host layout is:

```text
/nvme_zone3/home/ekamai1/surya/chimera/repos
/nvme_zone3/home/ekamai1/surya/chimera/data
```

The manager mounts those paths at `/workspace/repos` and
`/datasets/megadata`. Copy `examples/chimera/cluster.env.example` outside the
Git checkout, set one explicit `RUN_NAME`, and use the same configuration for
`preflight`, `launch`, `status`, `logs`, `stop`, and `cleanup`.

The image's `/workspace/load_env.sh` contains escaped quotes and must not be
sourced by the cluster flow. The manager explicitly sets:

```text
VIRTUAL_ENV=/workspace/venv
PATH=/workspace/venv/bin:$PATH
HF_HOME=/datasets/megadata/cache/huggingface
PYTHONPATH=/workspace/repos/Megatron-LM:/workspace/repos/Megatron-Bridge/src:/workspace/repos/transformers/src
```

Node list order defines ranks and the first node is the master. The manager
creates one shared run directory, checks topology and batch divisibility,
validates the image and Chimera imports on every node, and launches one
detached training container per node. It never clones or updates code during
launch. SFT and SimPO remain new fine-tuning stages and do not restore their
own optimizer or RNG state; only pretraining supports checkpoint resume. A
CPU-only controller can run `image-check`, `preflight`, and `dry-run`, but
`launch` requires GPU-enabled nodes.

## Fresh Container Setup

Expected container paths:

```text
/workspace/venv
/workspace/repos
```

Use `/datasets/megadata` for large artifacts inside cluster containers:

```bash
df -h
export DATA_ROOT=/datasets/megadata/chimera_smoke
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

SOURCE=$(realpath /workspace/repos/transformers/src/transformers)
if [ "$(realpath "$SITE")" != "$SOURCE" ]; then
  cp -a "$SOURCE/." "$SITE/"
fi
```

Do not rely on `rsync` in this container; the validated image did not include it.

Validate:

```bash
$PYTHON - <<'PY'
from transformers import ChimeraConfig, ChimeraForCausalLM
cfg = ChimeraConfig()
assert (cfg.first_k_dense_replace, cfg.last_k_dense_replace) == (2, 0)
assert (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (32, 4, 2048)
assert cfg.n_shared_experts == 0 and cfg.shared_expert_intermediate_size == 0
assert cfg.qk_layernorm and cfg.max_position_embeddings == 8192
assert cfg.rms_norm_eps == 1e-5
assert cfg.rope_parameters["factor"] == 1.0
assert cfg.router_load_balancing_type == "quantile_balancing"
assert (cfg.moe_qb_num_bins, cfg.moe_qb_ema_decay) == (1000, 0.0)
assert cfg.load_with_bias is True
print("chimera_transformers_ok", cfg.model_type, cfg.first_k_dense_replace, cfg.last_k_dense_replace)
PY
```

Optional Bridge unit test:

```bash
cd /workspace/repos/Megatron-Bridge
$PYTHON -m pytest -q tests/unit_tests/models/chimera/test_chimera_bridge.py
```

Validated result after the architecture migration: `12 passed`.

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
assert (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (32, 4, 2048)
assert cfg.n_shared_experts == 0 and cfg.shared_expert_intermediate_size == 0
assert cfg.qk_layernorm and cfg.max_position_embeddings == 8192
assert cfg.rms_norm_eps == 1e-5
assert cfg.rope_parameters["factor"] == 1.0
assert cfg.router_load_balancing_type == "quantile_balancing"
assert (cfg.moe_qb_num_bins, cfg.moe_qb_ema_decay) == (1000, 0.0)
assert cfg.load_with_bias is True
assert tok.convert_tokens_to_ids("<start_of_turn>") == 2
assert tok.convert_tokens_to_ids("<end_of_turn>") == 3
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
n_routed_experts=32
num_experts_per_tok=4
moe_intermediate_size=2048
n_shared_experts=0
qk_layernorm=true
max_position_embeddings=8192
YaRN factor=1
load_with_bias=true
<start_of_turn> id=2
<end_of_turn> id=3
<DUMMY_2> through <DUMMY_9> reserved at ids 4 through 11
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

The import preflight validates the complete HF architecture and exact key set.
For both conversion cycles plus exact per-key tensor/hash reports, run:

```bash
bash examples/chimera/verify_conversion.sh \
  --hf-source "$HF_RANDOM_FULL" \
  --work-dir "$DATA_ROOT/exact_conversion" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"
```

## Pretraining Data Format

Pretraining JSONL is raw text only: each row is `{"text": "..."}`. Parquet
input uses the same `text` column. `preprocess.sh` accepts one file, a glob, or
a directory recursively containing `.jsonl` and `.parquet` files, reports both
file and document counts, and combines them into one indexed dataset. Progress
logs include percentage, throughput, elapsed time, and ETA. Do not prepend
BOS. It passes `--append-eod`, so each document is followed by `<EOS>`.

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

The committed `train.sh` accepts smoke-only schedule overrides through the
environment; do not edit the canonical launcher.

Use 400 iterations for the two-document smoke. A 200-iteration run can show low
teacher-forced loss while the bare A and B prefixes still tie on their first
continuation token, causing one document to generate the other.

Run:

```bash
export RUNS_ROOT=$DATA_ROOT/runs
cd "$MEGATRON_LM"

TRAIN_DATA_PATH="$DATA_PREFIX" \
TOKENIZER_MODEL="$HF_REFERENCE" \
RUNS_ROOT="$RUNS_ROOT" \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=2 \
SEQ_LENGTH=512 \
TRAIN_ITERS=400 \
LR=1e-3 \
MIN_LR=1e-4 \
LR_DECAY_STYLE=cosine \
LR_WARMUP_ITERS=0 \
WEIGHT_DECAY=0.0 \
SAVE_INTERVAL=400 \
TP_SIZE=1 PP_SIZE=1 EP_SIZE=1 CP_SIZE=1 \
bash examples/chimera/train.sh
```

With two visible GPUs this is DP=2. If it fails specifically from GPU memory,
retry first with `MAIN_GRADS_DTYPE=bf16 EXP_AVG_DTYPE=bf16
EXP_AVG_SQ_DTYPE=bf16`. If that still fails, use exactly one fallback:
`EP_SIZE=2` or `TP_SIZE=2`. Keep the other size at 1 and record the OOM and
selected fallback in the validation report.

Capture the run paths dynamically:

```bash
export RUN_DIR=$(ls -td "$RUNS_ROOT"/* | head -n 1)
export CHECKPOINT_DIR=$RUN_DIR/checkpoints
echo "$RUN_DIR"
```

Validated 2xH200 result:

```text
iteration 1 lm loss:   1.133014E+01
iteration 400 lm loss: 2.168084E-01
skipped iterations: 0
nan iterations: 0
checkpoint: checkpoints/iter_0000400
checkpoint size: about 19G
```

No source restoration is needed because smoke settings are environment-only.
Verify `git status --short` remains unchanged after the run.

## Export To HF

Training now writes a validated, effective `run_config.yaml` at checkpoint
root. `examples/chimera/export.sh` refuses to invent metadata for a checkpoint;
it validates that file and copies it to the selected iteration when needed:

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
Using Bridge run config: <checkpoints>/iter_0000400/run_config.yaml
Successfully exported model to: <hf_export>
```

`examples/chimera/export.sh` copies the finalized tokenizer, chat template,
generation config, and tokenizer training report from the HF reference into the
export directory after Bridge writes the weights and model config.

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
assert (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (32, 4, 2048)
assert cfg.n_shared_experts == 0 and cfg.shared_expert_intermediate_size == 0
assert cfg.qk_layernorm and cfg.max_position_embeddings == 8192
assert cfg.rms_norm_eps == 1e-5
assert cfg.rope_parameters["factor"] == 1.0
assert cfg.load_with_bias is True
assert len(tok) == 50176
assert tok.convert_tokens_to_ids("<start_of_turn>") == 2
assert tok.convert_tokens_to_ids("<end_of_turn>") == 3
assert tok.chat_template
print("hf_export_ok", cfg.model_type, cfg.architectures, cfg.num_hidden_layers, len(tok), bool(tok.chat_template))
PY
```

## Verify Inference

```bash
cd "$MEGATRON_LM"
$PYTHON examples/chimera/verify_pretrain.py \
  --hf-model "$HF_EXPORT" \
  --prompt "CHIMERA_OVERFIT_KEY_A:" \
  --expected " The quiet engineer packed a silver notebook before sunrise and wrote down every signal from the training run.<EOS>"

$PYTHON examples/chimera/verify_pretrain.py \
  --hf-model "$HF_EXPORT" \
  --prompt "CHIMERA_OVERFIT_KEY_B:" \
  --expected " A careful researcher traced the river path through the valley and marked each bridge with a blue lantern.<EOS>"
```

Validated output:

```text
CHIMERA_OVERFIT_KEY_A: The quiet engineer packed a silver notebook before sunrise and wrote down every signal from the training run.<EOS>
Pretraining verification passed.
CHIMERA_OVERFIT_KEY_B: A careful researcher traced the river path through the valley and marked each bridge with a blue lantern.<EOS>
Pretraining verification passed.
```

## Chat Tokens And Template

The Chimera tokenizer uses two non-reasoning, non-tool chat markers:

```text
<start_of_turn> id=2
<end_of_turn>   id=3
```

They replace `<DUMMY_0>` and `<DUMMY_1>` while keeping tokenizer length exactly `50176`.
`<DUMMY_2>` through `<DUMMY_9>` remain reserved at IDs `4` through `11`. There is no `<unk>` token.
`<BOS>` remains ID `0` for compatibility but the chat template and Chimera SFT format do not insert it.
`<EOS>` remains ID `1` and is used for pretraining document separation and padding.

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

SFT JSONL rows use `messages` and are read directly by `SFTTokenizer`. Do not
run Megatron preprocessing for SFT. Production runs pack samples by default;
set `PACK_SAMPLES=false` for this exact-response smoke. The Chimera prompt
format masks system, user, and assistant header tokens, and trains only
assistant content plus `<end_of_turn>`.

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
PACK_SAMPLES=false \
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
iteration 1 lm loss:   1.747544E+01
iteration 120 lm loss: 6.252252E-05
checkpoint: <sft_run>/checkpoints/iter_0000120
```

Capture the SFT checkpoint path:

```bash
export SFT_RUN_DIR=$(ls -td "$CHAT_ROOT/sft_runs"/* | head -n 1)
export SFT_CHECKPOINT_DIR=$SFT_RUN_DIR/checkpoints
echo "$SFT_CHECKPOINT_DIR"
```

Export the SFT checkpoint to HF:

```bash
export HF_SFT_EXPORT=$CHAT_ROOT/hf_sft_export

cd "$MEGATRON_LM"
rm -rf "$HF_SFT_EXPORT"

bash examples/chimera/export.sh \
  --hf-reference "$HF_REFERENCE" \
  --mcore-path "$SFT_CHECKPOINT_DIR" \
  --hf-path "$HF_SFT_EXPORT" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"
```

Validate raw SFT inference with visible special tokens:

```bash
$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SFT_EXPORT" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SFT key A response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto

$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SFT_EXPORT" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SFT key B response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto
```

Raw inference should keep special tokens visible:

```text
CHIMERA_SFT_RESPONSE_A: jade lanterns align under quiet stars.<end_of_turn>
CHIMERA_SFT_RESPONSE_B: silver rivers circle patient mountains.<end_of_turn>
```

For custom Transformers inference, request a dictionary and pass its tensors
to `generate()` explicitly. Passing the complete `BatchEncoding` as positional
`inputs` fails because generation expects a tensor with a `shape`:

```python
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.environ["HF_SFT_EXPORT"]
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

## SimPO Smoke

SimPO JSONL rows use `chosen` and `rejected`, each as a messages list, and are
read directly by `SFTTokenizer`. Do not run Megatron preprocessing for SimPO.
Production runs pack samples by default; set `PACK_SAMPLES=false` for this
retention smoke.

`SimPODataset` reports the physical row count and requested logical sample
count, then wraps over the physical rows until the requested training length is
satisfied. Keep exactly two unique examples; do not duplicate them to match the
iteration count.

Create a two-example dataset:

```bash
export SIMPO_DATA=$CHAT_ROOT/data/simpo.jsonl
cat > "$SIMPO_DATA" <<'JSONL'
{"chosen":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key C response?"},{"role":"assistant","content":"CHIMERA_SIMPO_CHOSEN_C: amber maps reward careful answers."}],"rejected":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key C response?"},{"role":"assistant","content":"This is the rejected answer for C."}]}
{"chosen":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key D response?"},{"role":"assistant","content":"CHIMERA_SIMPO_CHOSEN_D: violet signals favor steady choices."}],"rejected":[{"role":"system","content":"You answer with the exact requested phrase."},{"role":"user","content":"What is the Chimera SimPO key D response?"},{"role":"assistant","content":"This is the rejected answer for D."}]}
JSONL
test "$(wc -l < "$SIMPO_DATA")" -eq 2
```

Run from the SFT MCore checkpoint:

```bash
cd "$MEGATRON_LM"
DATA_PATH="$SIMPO_DATA" \
TOKENIZER_MODEL="$HF_REFERENCE" \
MCORE_PATH="$SFT_CHECKPOINT_DIR" \
RUNS_ROOT="$CHAT_ROOT/simpo_runs" \
PACK_SAMPLES=false \
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
iteration 1 simpo loss:  1.755045E+01
iteration 40 simpo loss: 3.989361E-02
iteration 40 rewards/accuracies: 1.000000E+00
checkpoint: <simpo_run>/checkpoints/iter_0000040
```

Capture the SimPO checkpoint path:

```bash
export SIMPO_RUN_DIR=$(ls -td "$CHAT_ROOT/simpo_runs"/* | head -n 1)
export SIMPO_CHECKPOINT_DIR=$SIMPO_RUN_DIR/checkpoints
echo "$SIMPO_CHECKPOINT_DIR"
```

Export the SimPO checkpoint to HF:

```bash
export HF_SIMPO_EXPORT=$CHAT_ROOT/hf_simpo_export

cd "$MEGATRON_LM"
rm -rf "$HF_SIMPO_EXPORT"

bash examples/chimera/export.sh \
  --hf-reference "$HF_REFERENCE" \
  --mcore-path "$SIMPO_CHECKPOINT_DIR" \
  --hf-path "$HF_SIMPO_EXPORT" \
  --bridge-path "$MEGATRON_BRIDGE" \
  --python "$PYTHON"
```

Validate raw SimPO inference with visible special tokens:

```bash
$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SIMPO_EXPORT" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SimPO key C response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto

$PYTHON "$TRANSFORMERS/src/transformers/models/chimera/scripts/infer.py" \
  --model "$HF_SIMPO_EXPORT" \
  --chat \
  --system-prompt "You answer with the exact requested phrase." \
  --prompt "What is the Chimera SimPO key D response?" \
  --show-special-tokens \
  --max-new-tokens 48 \
  --device-map auto
```

Raw inference should keep special tokens visible:

```text
CHIMERA_SIMPO_CHOSEN_C: amber maps reward careful answers.<end_of_turn>
CHIMERA_SIMPO_CHOSEN_D: violet signals favor steady choices.<end_of_turn>
```

## Hard Rules

- Do not add BOS to pretraining JSONL.
- Pretraining document boundaries come from `--append-eod`.
- SFT and SimPO read JSONL directly through `SFTTokenizer`; do not run `preprocess.sh` for them.
- SFT and SimPO smoke runs are unpacked; do not pass `--pack-samples`.
- Keep real pretraining checkpoints resumable; use `--no-save-optim` and `--no-save-rng` only for short smoke runs.
- Do not write large artifacts to root overlay if persistent storage exists.
- Keep smoke schedule changes in environment variables; do not edit `train.sh` in place.
- If export fails with missing architecture, regenerate HF reference from a Transformers commit where no-weight export writes `architectures=["ChimeraForCausalLM"]`.
- If export reports missing architecture metadata, require the checkpoint-generated
  `run_config.yaml` at root, validate it with `architecture_contract.py`, and let
  `export.sh` copy that validated file to the latest `iter_*`. Never substitute
  the static template silently.
