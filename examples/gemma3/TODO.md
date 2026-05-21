# Gemma3 SFT (Supervised Fine-Tuning) Pipeline Plan

## Objective
Implement an end-to-end Supervised Fine-Tuning (SFT) pipeline for Gemma3 using Megatron-LM's built-in `SFTDataset` capabilities, allowing the model to be instruction-tuned on conversational data.

## Scope & Impact
- **Repository:** `Megatron-LM`
- **Impact:** Adds new automation scripts for dataset generation, SFT training, and verification. Does not alter the core CPT/Pretraining pipeline but extends it for instruction-tuning.

## Implementation Steps

### 1. SFT Dataset Preparation
**File:** `examples/gemma3/prepare_sft_data.py` (New)
- **Change:** Create a Python script to generate a `mock_sft.jsonl` file.
- **Format:** The JSONL must adhere to `SFTDataset` requirements, containing a `"messages"` list per line with `{"role": "...", "content": "..."}` dicts.
- **Content:** We will use the `medical_sample.txt` concept to create a few Q&A pairs (e.g., system prompt = "You are a medical assistant", user = "Patient presents with...", assistant = "Differential diagnosis includes...").

### 2. SFT Training Script
**File:** `examples/gemma3/04_sft_mcore.sh` (New)
- **Change:** Clone `02_train_mcore.sh` logic but adapt it for SFT.
- **Details:**
  - Add `--sft` flag to trigger `SFTDataset` loading in `pretrain_gpt.py`.
  - Provide `--train-data-path mock_sft.jsonl`.
  - Use `--tokenizer-type HuggingFaceTokenizer` and `--tokenizer-model google/gemma-3-1b-it` (instruction-tuned variant tokenizer) alongside `--sft-tokenizer-prompt-format default` to ensure the official Gemma3 chat template is applied by `SFTDataset`.
  - Lower the learning rate (e.g., `5e-6`) typical for SFT.

### 3. Automated SFT Pipeline Script
**File:** `examples/gemma3/run_1b_sft_pipeline.sh` (New)
- **Change:** Create an E2E orchestration script.
- **Details:**
  - Accept `HF_TOKEN`.
  - Run `01_convert_mcore.sh` (Import).
  - Run `prepare_sft_data.py` to generate the JSONL.
  - Run `04_sft_mcore.sh` (Training).
  - Run `03_export_mcore.sh` (Export).
  - Execute a final Python verification script that runs inference on the exported model using a user prompt from the dataset, verifying that the model outputs the trained assistant response.

### 4. SFT Runbook
**File:** `GEMMA3_SFT_RUNBOOK.md` (New)
- **Change:** Add documentation outlining the SFT steps, prerequisites, and success criteria.

## Verification & Testing
- Execute `run_1b_sft_pipeline.sh` on the remote Hopper node.
- Verify that the loss decreases over SFT iterations.
- Verify that the exported HF model correctly follows instructions by answering the medical prompt successfully in the specific format provided during SFT.
