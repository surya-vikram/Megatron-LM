# PubMed Data Preparation for Gemma 3 CPT

This guide details the end-to-end pipeline for preparing a high-fidelity medical corpus (PubMed) for Continual Pre-training (CPT) using the Gemma 3 stack.

## 1. Environment Setup
Ensure your production environment is active.
```bash
source /home/jovyan/load_env.sh
```

## 2. Data Ingestion & Splitting
The `ingest_pubmed.py` utility streams the official `ncbi/pubmed` baseline, cleans the text, deduplicates based on titles, and creates a deterministic Train/Val split.

### Command:
```bash
# --limit 0 pulls the entire 36M+ corpus. 
# Use --limit 1000000 for a quick verification run.
python3 examples/gemma3/utils/ingest_pubmed.py \
    --limit 0 \
    --output-prefix /home/jovyan/data/pubmed \
    --val-ratio 0.01
```

### Outputs:
- `/home/jovyan/data/pubmed_train.jsonl` (99% of data)
- `/home/jovyan/data/pubmed_val.jsonl` (1% of data)

---

## 3. Tokenization & Binary Packing
Megatron-LM requires data in a high-speed binary format (`.bin` and `.idx`). You must run this for both splits.

### A. Preprocess Training Set
```bash
bash examples/gemma3/preprocess.sh \
    --mode cpt \
    --input /home/jovyan/data/pubmed_train.jsonl \
    --output-prefix /home/jovyan/data/pubmed_train \
    --hf-tokenizer /home/jovyan/models/gemma-3-4b-pt \
    --workers 32
```

### B. Preprocess Validation Set
```bash
bash examples/gemma3/preprocess.sh \
    --mode cpt \
    --input /home/jovyan/data/pubmed_val.jsonl \
    --output-prefix /home/jovyan/data/pubmed_val \
    --hf-tokenizer /home/jovyan/models/gemma-3-4b-pt \
    --workers 32
```

---

## 4. Verification
Verify that the binary files exist and have non-zero sizes.
```bash
ls -lh /home/jovyan/data/pubmed_train_text_document.bin
ls -lh /home/jovyan/data/pubmed_val_text_document.bin
```

---

## 5. Launch Training
With the binary data ready, you can launch the production CPT run using `train.sh`.

```bash
bash examples/gemma3/train.sh \
    --mode cpt \
    --model-size 4b \
    --mcore-path /home/jovyan/models/gemma-3-4b-pt-mcore \
    --data-path /home/jovyan/data/pubmed_train_text_document \
    --valid-data-path /home/jovyan/data/pubmed_val_text_document \
    --wandb-project gemma3-medical-cpt \
    --iters 100000 \
    --lr 5e-7
```
