# Gemma 3 4B/12B Text Backbone Extraction Runbook

This runbook describes how to extract the text backbone from the Gemma 3 4B/12B multimodal models, train it in Megatron-Core, and export it back to HuggingFace.

## 1. Extraction (HF Multimodal -> Megatron Text)

Extract the language model component from the HF multimodal model and convert it to Megatron format.

```bash
python examples/gemma3/extract_text_backbone.py \
    --hf-model google/gemma-3-4b-it \
    --save-path ./checkpoints/gemma3-4b-text-mcore \
    --tp-size 1 \
    --pp-size 1
```

## 2. Training in Megatron-Core

Use the existing `02_train_mcore.sh` script to train or verify the extracted checkpoint.

```bash
./examples/gemma3/02_train_mcore.sh \
    ./checkpoints/gemma3-4b-text-mcore \
    4B
```

*Note: Ensure `MODEL_SIZE` is set to `4B` or `12B` correctly in the script.*

## 3. Export (Megatron Text -> HF)

### Option A: Standalone Text Model

Export the trained MCore checkpoint as a standalone `Gemma3ForCausalLM` model.

```bash
python examples/gemma3/export_standalone_text.py \
    --megatron-path ./checkpoints/gemma3-4b-text-mcore_trained \
    --hf-save-path ./hf_exports/gemma3-4b-standalone-text \
    --hf-tokenizer-path google/gemma-3-4b-it
```

### Option B: Stitched Multimodal Model

Stitch the trained text weights back into the original multimodal architecture.

```bash
python examples/gemma3/export_stitched_multimodal.py \
    --megatron-path ./checkpoints/gemma3-4b-text-mcore_trained \
    --vlm-hf-path google/gemma-3-4b-it \
    --output-path ./hf_exports/gemma3-4b-stitched-vlm
```

## 4. Verification

### Verify Standalone Parity
Compare logits between the original multimodal model (text-only pass) and the standalone text model.

```bash
python examples/gemma3/verify_standalone_text.py \
    --vlm-path google/gemma-3-4b-it \
    --text-path ./hf_exports/gemma3-4b-standalone-text
```

### Verify Stitched Parity
Compare logits between the original multimodal model and the stitched multimodal model.

```bash
# (Implementation of verify_stitched_multimodal.py is recommended for image-text parity)
```
