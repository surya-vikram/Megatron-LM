# Gemma 3 Production Training Runbook

This runbook describes the end-to-end workflow for importing, training (CPT → SFT → SimPO), and exporting Gemma 3 models using Megatron-LM.

---

## 1. Environment Setup

You can choose between a self-contained container or a bare-metal bootstrap.

### Option A: Self-Contained Container Setup
(See Docker workflow in repository root). Mount your datasets to the centralized root:
```bash
docker run --gpus all -it --rm \
  -v /path/to/dataset:/datasets/megadata \
  -v /path/to/repos:/workspace/repos \
  suryavikram6/megatron-gemma:latest
```

---

### Option B: Bare-Metal VM Setup
```bash
bash examples/gemma3/setup.sh
source /datasets/megadata/load_env.sh
```

---

## 2. Import: HuggingFace → Megatron

Convert the original Google HF checkpoint into a Megatron-Core checkpoint.
```bash
bash examples/gemma3/import.sh \
    --hf-model /datasets/megadata/hf_models/gemma-3-4b-pt \
    --mcore-path /datasets/megadata/mcore_models/gemma-3-4b-pt-mcore
```

---

## 3. Preprocess: Prepare Training Data

### CPT — JSONL to Megatron Binary
Megatron requires JSONL input (one `{"text": "..."}` object per line).
```bash
bash examples/gemma3/preprocess.sh \
    --mode cpt \
    --input /datasets/megadata/cpt/corpus.jsonl \
    --output-prefix /datasets/megadata/cpt/corpus_bin \
    --hf-tokenizer /datasets/megadata/hf_models/gemma-3-4b-pt
```
> **Note:** Pass `--data-path` to `train.sh` **without** the `.bin`/`.idx` extension.  
> Correct: `--data-path /datasets/megadata/cpt/corpus_bin_text_document`

---

## 4. Train

`train.sh` uses a centralized root variable `MEGADATA_ROOT="/datasets/megadata"`. All outputs are saved in timestamped directories under `training_runs/` to prevent overwrites.

### Mandatory Arguments (Tier 1)
You **must** explicitly provide these paths:
*   `--mcore-path`: Path to Megatron weights.
*   `--tokenizer-model`: Path to HF tokenizer directory.
*   `--data-path`: Path to training data.

### Minimal Commands

#### Continual Pre-Training (CPT)
```bash
bash examples/gemma3/train.sh \
    --mode cpt \
    --model-size 4b \
    --mcore-path /datasets/megadata/mcore_models/gemma-3-4b-pt-mcore \
    --tokenizer-model /datasets/megadata/hf_models/gemma-3-4b-pt \
    --data-path /datasets/megadata/cpt/mock_cpt_text_document
```

#### Supervised Fine-Tuning (SFT)
```bash
bash examples/gemma3/train.sh \
    --mode sft \
    --model-size 4b \
    --mcore-path /datasets/megadata/mcore_models/gemma-3-4b-pt-mcore \
    --tokenizer-model /datasets/megadata/hf_models/gemma-3-4b-pt \
    --data-path /datasets/megadata/sft/mock_sft.jsonl
```

#### Preference Optimization (SimPO)
```bash
bash examples/gemma3/train.sh \
    --mode simpo \
    --model-size 4b \
    --mcore-path /datasets/megadata/mcore_models/gemma-3-4b-pt-mcore \
    --tokenizer-model /datasets/megadata/hf_models/gemma-3-4b-pt \
    --data-path /datasets/megadata/preference/mock_simpo.jsonl
```

---

## 5. Output Isolation

Checkpoints and logs are automatically isolated per run:
*   **Checkpoints**: `/datasets/megadata/training_runs/YYYYMMDD_HHMMSS/checkpoints`
*   **Logs**: `/datasets/megadata/training_runs/YYYYMMDD_HHMMSS/logs/tb`

---
## 6. Export: Megatron → HuggingFace

```bash
# Standard export (TP=1)
bash examples/gemma3/export.sh \
    --target text \
    --tp-size 1 \
    --mcore-path /datasets/megadata/training_runs/YOUR_TIMESTAMP/checkpoints \
    --hf-reference /datasets/megadata/hf_models/gemma-3-4b-pt \
    --save-path /datasets/megadata/hf_models/gemma-3-4b-sft-hf

# For TP=2 (e.g., 12B models), pass --tp-size 2
```
