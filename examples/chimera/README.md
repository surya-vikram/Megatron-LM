# Chimera Megatron-LM Flow

This example trains Chimera from a Hugging Face checkpoint converted through
Megatron-Bridge, then exports the trained Megatron-Core checkpoint back to HF.

Default remote paths:

```text
HF reference:        /datasets/megadata/hf_models/chimera-12b
MCore import:        /datasets/megadata/chimera_bridge_validation/megatron_import
Overfit data:        /datasets/megadata/chimera/overfit_doc_text_document
Training checkpoint: /datasets/megadata/chimera_runs/overfit/checkpoints
HF export:           /datasets/megadata/hf_exports/chimera-overfit-hf
Megatron-Bridge:     /workspace/repos/Megatron-Bridge
```

## 1. Import HF to MCore

Skip this if `/datasets/megadata/chimera_bridge_validation/megatron_import`
already exists.

```bash
bash examples/chimera/import.sh
```

## 2. Preprocess Overfit Document

```bash
bash examples/chimera/preprocess.sh
```

Expected output:

```text
/datasets/megadata/chimera/overfit_doc_text_document.bin
/datasets/megadata/chimera/overfit_doc_text_document.idx
```

Use the printed data prefix as `--data-path` for training.

## 3. Overfit Train

```bash
bash examples/chimera/train.sh
```

Useful overrides:

```bash
bash examples/chimera/train.sh \
  --data-path /datasets/megadata/chimera/overfit_doc_text_document \
  --mcore-path /datasets/megadata/chimera_bridge_validation/megatron_import \
  --save-path /datasets/megadata/chimera_runs/overfit/checkpoints \
  --train-iters 100 \
  --seq-length 512 \
  --lr 2e-4
```

Recommended two-GPU overfit run:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/chimera/train.sh \
  --gpus-per-node 2 \
  --tp-size 2 \
  --ep-size 2 \
  --expert-tp-size 1 \
  --global-batch-size 1 \
  --train-iters 20 \
  --save-interval 20 \
  --seq-length 128 \
  --lr 1e-3 \
  --min-lr 1e-4
```

Single-GPU fallback is supported with `--cpu-offload`, but it is slow and
should not be used when two GPUs are available.

Validation points:

- checkpoint loads without shape mismatch
- training loss decreases
- final checkpoint is written under `--save-path`

## 4. Export MCore to HF

```bash
bash examples/chimera/export.sh
```

## 5. Verify Completion

```bash
python3 examples/chimera/verify_completion.py \
  --hf-model /datasets/megadata/hf_exports/chimera-overfit-hf
```

The verifier checks whether greedy generation from the overfit key prompt
contains the expected memorized phrase.
