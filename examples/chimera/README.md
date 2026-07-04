# Chimera Megatron-LM Flow

This example pretrains Chimera from random initialization with Megatron-LM and
exports trained checkpoints back to Hugging Face through Megatron-Bridge.

## Locked Architecture

```text
layers:              25
dense layers:        first 2
MoE layers:          remaining 23
Megatron pattern:    [0]*2+[1]*23
HF dense config:     first_k_dense_replace=2, last_k_dense_replace=0
baseline parallel:   TP=1 PP=1 EP=1 ETP=1 CP=1
```

## Default Paths

```text
HF/tokenizer reference: /datasets/megadata/hf_models/chimera-10b
Preprocessed data:     /datasets/megadata/chimera/overfit_doc_text_document
Run root:              /datasets/megadata/chimera_runs
HF export:             /datasets/megadata/hf_exports/chimera-overfit-hf
Megatron-Bridge:       /workspace/repos/Megatron-Bridge
```

## 1. Create HF Reference Artifacts

Use the Transformers Chimera export script to create config and tokenizer
artifacts. Full random HF weights are optional; Megatron-LM training here starts
from random initialization.

```bash
cd /path/to/transformers
python3 src/transformers/models/chimera/scripts/export_to_hf.py \
  --output /datasets/megadata/hf_models/chimera-10b \
  --no-weights
```

## 2. Preprocess Documents

Input is JSONL with raw text:

```json
{"text":"sample A text..."}
{"text":"sample B text..."}
```

Run:

```bash
bash examples/chimera/preprocess.sh \
  --input examples/chimera/overfit_doc.jsonl \
  --output-prefix /datasets/megadata/chimera/overfit_doc \
  --tokenizer-model /datasets/megadata/hf_models/chimera-10b \
  --workers 8
```

The preprocessing script uses `--append-eod`, so Megatron appends the tokenizer
EOS/EOD token after each document. Do not manually add BOS to pretraining text.

Expected output:

```text
/datasets/megadata/chimera/overfit_doc_text_document.bin
/datasets/megadata/chimera/overfit_doc_text_document.idx
```

## 3. Random-Init Pretraining

Set the data prefix, tokenizer/reference directory, and run root:

```bash
DATA_PATH=/datasets/megadata/chimera/overfit_doc_text_document \
TOKENIZER_MODEL=/datasets/megadata/hf_models/chimera-10b \
RUNS_ROOT=/datasets/megadata/chimera_runs \
bash examples/chimera/train.sh
```

`train.sh` creates a timestamped IST run directory:

```text
/datasets/megadata/chimera_runs/YYYYMMDD_HHMMSS/
  checkpoints/
  data_cache/
  logs/train.log
  tensorboard/
  run_paths.env
  train.sh
```

The current baseline is 8k sequence pretraining with 32k YaRN metadata. Use the
saved checkpoint for later context extension instead of training 32k from the
first run.

## 4. Export MCore to HF

```bash
bash examples/chimera/export.sh \
  --hf-reference /datasets/megadata/hf_models/chimera-10b \
  --mcore-path /datasets/megadata/chimera_runs/YYYYMMDD_HHMMSS/checkpoints \
  --hf-path /datasets/megadata/hf_exports/chimera-overfit-hf \
  --bridge-path /workspace/repos/Megatron-Bridge
```

## 5. Verify Completion

```bash
python3 examples/chimera/verify_completion.py \
  --hf-model /datasets/megadata/hf_exports/chimera-overfit-hf
```

The verifier checks whether greedy generation from the overfit key prompt
contains the expected memorized phrase.
