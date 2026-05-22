# Gemma 3 Production Training Runbook (4B & 12B)

This runbook describes the modular workflow for importing, training, and exporting Gemma 3 models using Megatron-LM.

## 1. Environment Setup
Bootstrap the node with all dependencies (FlashAttn, TE, Bridge) and persistent symlinks.
```bash
bash examples/gemma3/setup.sh
source /home/jovyan/load_env.sh
```

## 2. Import: HF VLM -> Megatron
Convert the original Google VLM into a Megatron-Core text backbone.
```bash
bash examples/gemma3/import.sh \
    --hf-model /path/to/gemma-3-12b-pt \
    --mcore-path /path/to/gemma-3-12b-pt-mcore
```

## 3. Train: CPT or SFT
Launch unified training with Precision-Aware Adam and dynamic model detection.
### Continual Pre-training (CPT)
```bash
bash examples/gemma3/train.sh \
    --mode cpt \
    --model-size 12b \
    --hf-model /path/to/gemma-3-12b-pt \
    --mcore-path /path/to/gemma-3-12b-pt-mcore \
    --data-path /path/to/my_data_text_document \
    --save-path /path/to/checkpoints
```

### Instruction Tuning (SFT)
```bash
bash examples/gemma3/train.sh \
    --mode sft \
    --model-size 4b \
    --hf-model /path/to/gemma-3-4b-pt \
    --mcore-path /path/to/gemma-3-4b-pt-mcore \
    --data-path /path/to/my_instructions.jsonl \
    --save-path /path/to/checkpoints
```

## 4. Export: Megatron -> HF
Convert the trained weights back to the HuggingFace format.
### Standalone Text Model
```bash
bash examples/gemma3/export.sh \
    --target text \
    --mcore-path /path/to/trained_mcore \
    --hf-reference /path/to/gemma-3-pt \
    --save-path /path/to/exported_hf
```

### Stitched Multimodal VLM
```bash
bash examples/gemma3/export.sh \
    --target vlm \
    --mcore-path /path/to/trained_mcore \
    --hf-reference /path/to/gemma-3-pt \
    --save-path /path/to/stitched_vlm
```

---
**Note for 12B:** 12B requires TP=2 or PP=2 for production runs to avoid OOM on single-GPU (even H200). Update `train.sh` distributed flags accordingly for multi-GPU nodes.
