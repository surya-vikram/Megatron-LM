# Gemma3 SFT (Supervised Fine-Tuning) Pipeline Plan

## Objective
Implement an end-to-end Supervised Fine-Tuning (SFT) pipeline for Gemma3 using Megatron-LM's built-in `SFTDataset` capabilities, utilizing the high-quality **LDJnr/Capybara** dataset from HuggingFace to verify model learning and instruction-following.

## Scope & Impact
- **Repository:** `Megatron-LM`
- **Impact:** Adds automation for real-data SFT, specifically targeting the transition from "completion" to "assistant" behavior using multi-turn conversational data.

## Implementation Steps

### 1. SFT Dataset Preparation
**File:** `examples/gemma3/prepare_sft_data.py`
- **Change:** Create a script that uses the `datasets` library to pull the `LDJnr/Capybara` dataset.
- **Processing:** 
    - Convert the Capybara format to the Megatron `SFTDataset` JSONL format (`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`).
    - Create two files:
        1. `medical_gold_overfit.jsonl`: 5 high-quality medical Q&A pairs (manually injected to verify memorization).
        2. `capybara_sft_subset.jsonl`: ~500-1000 samples from Capybara for general instruction tuning.

### 2. SFT Training Script
**File:** `examples/gemma3/04_sft_mcore.sh`
- **Configuration:**
    - Flag: `--sft`
    - Tokenizer: `google/gemma-3-1b-it` (to inherit the `<start_of_turn>` template).
    - Data: `--train-data-path capybara_sft_subset.jsonl`.
    - Hyperparameters: LR `5e-6`, Global Batch Size `8`, Seq Length `1024`.

### 3. Automated Verification Script
**File:** `examples/gemma3/verify_sft_results.py` (New)
- **Tests:**
    - **Test A (Memorization):** Prompt with a "Gold" medical question. Success if output matches the injected answer > 80% (proving weight update).
    - **Test B (Format & Style):** Prompt with a novel question. Success if the model correctly wraps output in `<start_of_turn>model` format and adopts the helpful assistant persona characteristic of the Capybara dataset.

### 4. Orchestration Pipeline
**File:** `examples/gemma3/run_1b_sft_pipeline.sh`
- **Flow:** `Import HF` -> `Prepare Data (Capybara + Gold)` -> `Train (SFT)` -> `Export to HF` -> `Run verify_sft_results.py`.

## Verification & Testing
- Monitor loss during SFT; expect a significant drop as the model aligns with the instruction format.
- Final validation: The exported model must pass `verify_sft_results.py` with 100% success on formatting and high alignment with the training distribution.
