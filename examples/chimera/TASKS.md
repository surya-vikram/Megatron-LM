# Chimera Task Plan

## Locked Decisions

- Architecture: 25 decoder layers with first 2 dense layers and remaining 23 MoE layers.
- HF config fields: `first_k_dense_replace=2`, `last_k_dense_replace=0`.
- Megatron layer pattern: `--moe-layer-freq "[0]*2+[1]*23"`.
- Pretraining starts from Megatron-LM random initialization.
- Pretraining documents are raw JSONL text records; Megatron preprocessing appends EOS/EOD with `--append-eod`.
- No BOS token is added to pretraining documents.
- 8k sequence length is the baseline pretraining stage; 32k is represented by YaRN metadata and should be reached by later context extension.
- Default intra-document masking is off for the stable TE/CUDA-graph pretraining path.

## Milestones

### 1. HF Model Artifacts

- Keep bundled Chimera tokenizer artifacts.
- Ensure generated HF config, tokenizer, generation config, and README are self-contained.
- Keep vocab size at 50176.
- Token replacement and chat-template special tokens are deferred.

### 2. Megatron-LM Pretraining Smoke

- Preprocess two small JSONL text samples and inspect decoded token flow.
- Train from random initialization with the Chimera Megatron script after confirmation.
- Use high save/eval intervals for smoke tests to avoid checkpoint clutter.
- Export trained checkpoint to HF and verify memorized completion.

### 3. SFT Smoke

- Decide packed-sequence behavior before implementation.
- Verify two-sample SFT formatting and masking before training.

### 4. SimPO Smoke

- Decide packed-sequence behavior before implementation.
- Verify two-sample preference formatting and masking before training.

## Cross-Repo Consistency Targets

- Transformers Chimera defaults and export script use first 2 dense, last 0 dense.
- Megatron-Bridge Chimera tests verify `[0]*2+[1]*23` conversion both directions.
- Megatron-LM docs, skill, and training script describe random-init pretraining with the locked architecture.
