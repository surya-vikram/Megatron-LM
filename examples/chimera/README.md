# Chimera Megatron-LM Example

This directory contains the reproducible Chimera pretraining, SFT, SimPO, and
Megatron-Bridge conversion workflow. See [RUNBOOK.md](RUNBOOK.md) for commands.

## Architecture

```text
decoder layers:       25
dense layers:         first 2
MoE layers:           remaining 23
Megatron pattern:     [0]*2+[1]*23
HF dense fields:      first_k_dense_replace=2, last_k_dense_replace=0
vocabulary size:      50176
context:              8k training baseline, 32k YaRN model limit
```

The production baseline is TP=1, PP=1, EP=1, ETP=1, CP=1. The two-GPU smoke
configuration temporarily changes EP to 2.

## Data

```text
data/pretrain/overfit.jsonl                 raw {"text": ...} documents
data/pretrain/overfit_text_document.bin     preprocessed token data
data/pretrain/overfit_text_document.idx     preprocessed document index
data/sft/overfit.jsonl                      direct {"messages": [...]} rows
data/simpo/overfit.jsonl                    direct chosen/rejected rows
```

Pretraining preprocessing appends `<EOS>` to each document. It does not add
`<BOS>`. SFT and SimPO are read directly through `SFTTokenizer`; they are not
passed through `preprocess_data.py` and the example scripts do not enable
`--pack-samples`. `train.sh` keeps `INTRA_DOC_MASKING=false` by default, so
pretraining uses an ordinary causal mask across EOS-delimited documents.

## Files

- `preprocess.sh`: Chimera convenience wrapper around `tools/preprocess_data.py` that recursively converts pretraining `.jsonl` and parquet files to one Megatron `.bin/.idx` dataset, using the HF tokenizer and appending `<EOS>` to every document.
- `train.sh`: random-init Chimera pretraining.
- `sft.sh`: supervised fine-tuning from an MCore checkpoint.
- `simpo.sh`: SimPO preference tuning from an MCore checkpoint.
- `import.sh`: convert a complete HF checkpoint to MCore.
- `export.sh`: convert an MCore checkpoint to HF.
- `pretrain_chimera.py`: Megatron training entry point shared by all stages.
- `run_config.yaml`: architecture metadata consumed by Megatron-Bridge export.
- `verify_pretrain.py`: assert a plain-text pretraining overfit completion.

Large HF weights, checkpoints, optimizer state, logs, and generated runs belong
on persistent storage outside the Git checkout.
