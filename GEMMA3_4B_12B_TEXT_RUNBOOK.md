# Gemma 3 4B & 12B Text Backbone Extraction Runbook

This runbook describes the process for extracting, training, and verifying the text-only backbones for Gemma 3 4B and 12B models.

## 1. Overview
Gemma 3 4B and 12B models are natively multimodal (VLM). This pipeline extracts the `language_model` component into a standalone Megatron-Core checkpoint for efficient SFT/CPT, and provides tools to export it back to a HuggingFace `Gemma3ForCausalLM` model.

## 2. Key Features
- **Standalone Extraction:** Isolates text weights from vision components.
- **PT to IT Metadata Injection:** Automatically injects chat templates and special tokens when using an IT model as a reference during export. This allows Pre-trained (PT) weights to be "SFT-ready" with the correct chat formatting.
- **Fidelity Verification:** Automated logit parity checks between the VLM (text-only mode) and the extracted standalone model.

## 3. Extraction & Verification Pipeline

### 4B Model
Run the automated test for 4B:
```bash
bash remote_test_gemma3_4b.sh
```

### 12B Model
Run the automated test for 12B:
```bash
bash remote_test_gemma3_12b.sh
```

## 4. Manual Execution Steps

### Step 1: Extraction (HF -> Megatron)
```bash
python examples/gemma3/extract_text_backbone.py \
    --hf-model /path/to/gemma-3-12b-pt \
    --save-path /path/to/mcore-checkpoint \
    --tp-size 1 \
    --pp-size 1
```

### Step 2: Export (Megatron -> HF)
To ensure the exported model has chat templates (even for PT weights), point `--hf-tokenizer-path` to the **IT** version of the model.
```bash
python examples/gemma3/export_standalone_text.py \
    --megatron-path /path/to/mcore-checkpoint \
    --hf-save-path /path/to/standalone-hf \
    --hf-tokenizer-path /path/to/gemma-3-12b-it \
    --tp-size 1 \
    --pp-size 1
```

### Step 3: Verification
```bash
python examples/gemma3/verify_standalone_text.py \
    --vlm-path /path/to/gemma-3-12b-pt \
    --text-path /path/to/standalone-hf \
    --prompt "The capital of France is"
```

## 5. Troubleshooting
- **Storage Limits:** Use `/home/jovyan/models` for storage on remote pods to avoid ephemeral storage eviction.
- **OOM Errors:** The extraction script uses manual layer-by-layer loading to bypass deepcopy OOM issues. Ensure `load_weights=False` is maintained in the bridge provider initialization.
