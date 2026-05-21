# Gemma3 SFT (Supervised Fine-Tuning) Pipeline Plan

## Objective
Implement an end-to-end Supervised Fine-Tuning (SFT) pipeline for Gemma3 using Megatron-LM's built-in `SFTDataset` capabilities, utilizing the high-quality **LDJnr/Capybara** dataset from HuggingFace to verify model learning and instruction-following.

## Scope & Impact
- **Repository:** `Megatron-LM`
- **Impact:** Adds automation for real-data SFT, specifically targeting the transition from "completion" to "assistant" behavior using multi-turn conversational data.

## Implementation Steps

### 1. prepare_sft_data.py (The Data Architect)
- **Role:** Transforms raw internet data into Megatron-digestible instruction formats.
- **Actions:**
    - Downloads `LDJnr/Capybara` via `datasets` library.
    - Maps multi-turn conversations into `{"messages": [{"role": "user/assistant", "content": "..."}]}` JSONL format.
    - Injects **"Gold" Medical Samples**: Manually adds 5 specific Q&A pairs derived from `medical_sample.txt` to act as "canaries" for memorization proof.
- **Output:** `capybara_sft_subset.jsonl`.

### 2. 04_sft_mcore.sh (The Trainer)
- **Role:** Bash wrapper for GPU-accelerated SFT training.
- **Key Configs:**
    - `--sft`: Activates `SFTDataset` to mask user tokens and only calculate loss on assistant responses.
    - `--tokenizer-model google/gemma-3-1b-it`: Loads the official Gemma3 Jinja chat template to ensure correct `<start_of_turn>` wrapping.
    - **Hyperparameters**: LR `5e-6` (low for stability), Global Batch Size `8`, Seq Length `1024`.

### 3. verify_sft_results.py (The Judge)
- **Role:** Post-export validation of the fine-tuned HuggingFace model.
- **Tests:**
    - **Memorization Test**: Prompts with "Gold" medical questions. Success = >80% match with injected answers (verifies gradient update).
    - **Format & Style Test**: Prompts with novel questions. Success = Output starts with `<start_of_turn>model` and terminates correctly at `<end_of_turn>` (verifies template alignment).

### 4. run_1b_sft_pipeline.sh (The Orchestrator)
- **Flow:** `Import HF` -> `Prepare Data (Capybara + Gold)` -> `Train (SFT)` -> `Export to HF` -> `Run verify_sft_results.py`.

## Verification & Testing
- Monitor loss during SFT; expect a significant drop as the model aligns with the instruction format.
- Final validation: The exported model must pass `verify_sft_results.py` with 100% success on formatting and high alignment with the training distribution.
