# Chimera Context Extension

This document describes the canonical Chimera lifecycle: fresh 8K YaRN pretraining,
sequential 32K/64K/128K continued pretraining, mixed-length SFT at the final geometry,
and phase-preserving conversion to Transformers. Ordinary RoPE and no-position modes are
not supported by this workflow.

## Core concepts

Four values must not be conflated:

- `--seq-length` is the number of tokens processed in each training sequence.
- `--max-position-embeddings` is the configured upper bound and exported context metadata.
- `yarn_original_max_position_embeddings` is the context length before extension. For
  Chimera, it remains `8192` in every extension phase.
- `yarn_rotary_scaling_factor` changes the rotary frequencies used by the model.

For Chimera:

```text
YaRN factor = phase target context / 8192
```

RoPE does not contain a learned embedding row for every position. Raising
`max_position_embeddings` therefore does not teach the model to use those positions.
Long-context capability comes from training on long sequences with the corresponding
RoPE or YaRN configuration.

The phase resolver locks maximum, factor, and original context together. It rejects a
factor or maximum that belongs to another phase. A checkpoint therefore advertises only
the context phase on which it is being trained.

## Phase configuration

Use the following values for an original 8K context:

| Phase | `seq-length` | `max-position-embeddings` | YaRN factor | YaRN original |
| --- | ---: | ---: | ---: | ---: |
| Base pretraining | 8192 | 8192 | 1 | 8192 |
| 32K extension | 32768 | 32768 | 4 | 8192 |
| 64K extension | 65536 | 65536 | 8 | 8192 |
| 128K extension | 131072 | 131072 | 16 | 8192 |

The maximum and factor should describe the same phase. Do not, for example, combine a
32K maximum with factor `16`.

The production launcher deliberately enforces the complete sequence
`8K -> 32K -> 64K -> 128K`. Each transition validates the source checkpoint's explicit
phase metadata before loading weights.

## Canonical launch sequence

Start a new base model. Do not initialize this run from the legacy checkpoint that was
trained while TE attention CUDA graphs omitted rotary inputs.

```bash
CONTEXT_PHASE=8k \
TRAIN_DATA_PATH="$DATA_PREFIX" \
TOKENIZER_MODEL="$HF_8K_REFERENCE" \
RUNS_ROOT="$BASE_RUNS" \
bash examples/chimera/train.sh
```

After each phase passes its validation gates, launch the next phase from the checkpoint
root. `context_extend.sh` loads model weights only and starts fresh Adam optimizer, RNG,
iteration, and scheduler state.

```bash
CONTEXT_PHASE=32k LOAD_CHECKPOINT="$CKPT_8K" \
TRAIN_DATA_PATH="$LONG_DATA_PREFIX" TOKENIZER_MODEL="$HF_8K_REFERENCE" \
bash examples/chimera/context_extend.sh

CONTEXT_PHASE=64k LOAD_CHECKPOINT="$CKPT_32K" \
TRAIN_DATA_PATH="$LONG_DATA_PREFIX" TOKENIZER_MODEL="$HF_8K_REFERENCE" \
bash examples/chimera/context_extend.sh

CONTEXT_PHASE=128k LOAD_CHECKPOINT="$CKPT_64K" \
TRAIN_DATA_PATH="$LONG_DATA_PREFIX" TOKENIZER_MODEL="$HF_8K_REFERENCE" \
bash examples/chimera/context_extend.sh
```

The extension defaults preserve approximately 4.72M tokens per update, use a 10B-token
phase budget, cosine decay from `1e-5` to `1e-6`, 10% warmup, Adam, and weight decay
`0.1`. Override token budget or parallelism through environment variables, not tracked
script edits.

## Training arguments

### Base 8K phase

```bash
--seq-length 8192
--max-position-embeddings 8192
```

Use:

```text
yarn_rotary_scaling_factor = 1.0
yarn_original_max_position_embeddings = 8192
```

Factor `1` makes the YaRN interpolation path equivalent to unscaled rotary frequencies.

### 32K extension

```bash
--seq-length 32768
--max-position-embeddings 32768
```

Use factor `4.0` and retain the original maximum of `8192`.

### 64K extension

```bash
--seq-length 65536
--max-position-embeddings 65536
```

Use factor `8.0` and retain the original maximum of `8192`.

### 128K extension

```bash
--seq-length 131072
--max-position-embeddings 131072
```

Use factor `16.0` and retain the original maximum of `8192`.

## Batch and parallelism scaling

The current 8K recipe processes approximately 4.72 million tokens per optimizer update:

```text
8192 * 576 = 4,718,592 tokens/update
```

The following global batch sizes preserve that token batch:

| Context | Suggested MBS | GBS | Tokens/update |
| --- | ---: | ---: | ---: |
| 8K | 4 | 576 | 4,718,592 |
| 32K | 1 | 144 | 4,718,592 |
| 64K | 1 | 72 | 4,718,592 |
| 128K | 1 | 36 | 4,718,592 |

The selected GBS must be divisible by the effective data-parallel batch unit:

```text
DP size = world size / (TP size * PP size * CP size)
batch unit = DP size * MBS
```

Starting context-parallel settings on H200-class hardware are:

| Context | CP starting point |
| --- | ---: |
| 32K | 4 |
| 64K | 4 or 8 |
| 128K | 8 or 16 |

Start with `--cp-comm-type p2p`. Benchmark the smallest CP size that fits reliably and
provides good throughput. Recalculate GBS divisibility whenever world size or model
parallelism changes.

## Context-extension optimizer schedule

Treat context extension as a new, low-learning-rate continued-pretraining phase. A stable
starting configuration is:

```bash
--load "$LOAD_CHECKPOINT"
--finetune
--no-load-optim
--no-load-rng
--lr 1e-5
--min-lr 1e-6
--lr-decay-style cosine
--weight-decay 0.1
```

Remove the base phase's WSD-only arguments when using cosine decay:

```text
--lr-wsd-decay-style
--lr-wsd-decay-iters
```

Set warmup to approximately 10% of the extension phase. Calculate the number of
iterations from the desired token budget:

```text
train iters = extension tokens / (sequence length * global batch size)
```

For example, 10B tokens with a 4,718,592-token global update requires approximately
2120 iterations and 212 warmup iterations.

Using `--finetune --no-load-optim --no-load-rng` loads model weights while starting new
iteration counters, optimizer state, RNG state, and scheduler. This avoids restoring the
previous phase's completed schedule.

Use a new timestamped run directory and a fresh data cache for every context phase.

## Long-context data

The extension corpus must contain genuinely long, coherent documents. Filling a 128K
sequence entirely with unrelated short documents exposes the model to large position IDs,
but provides few meaningful long-range dependencies.

A practical extension mixture should contain:

- Long code repositories, books, papers, or technical documents.
- A domain-balanced portion of the original high-quality pretraining mixture.
- A separate held-out long-context validation corpus.
- Documents and packed sequences covering the target length, not only short examples.

If cross-document attention is enabled, EOD boundaries still identify documents but later
documents may attend to earlier unrelated documents. If intra-document masking is enabled,
the corpus must contain sufficiently long individual documents because position IDs and
attention restart at EOD boundaries.

## Implemented phase contract

### Megatron-LM

- `context_phase.sh` is the single 8K/32K/64K/128K resolver used by pretraining,
  Tiny Chimera, context extension, SFT, and SimPO.
- `context_extend.sh` validates the previous phase and launches weight-only continued
  pretraining with a fresh Adam schedule.
- `pretrain_chimera.py` writes `chimera_context_phase`, maximum, factor, original context,
  and fractional-correction metadata into checkpoint `run_config.yaml`.
- `sft.sh` and `simpo.sh` default to the final 128K/factor-16 geometry. Packing is enabled,
  Adam is the default optimizer, and SFT uses per-token loss normalization. A shorter
  `SEQ_LENGTH` is allowed while the published maximum and factor remain 128K/16.

### Transformers

- `ChimeraConfig` supports only the four canonical YaRN phases and rejects inconsistent
  maximum/factor/original/epsilon/theta values.
- `export_to_hf.py --context-phase {8k,32k,64k,128k}` emits phase-specific metadata.
- Transformers uses `truncate=false`, matching Megatron's
  `yarn_correction_range_round_to_int=false`.
- Raw `infer.py` tokenizes with `add_special_tokens=false`; chat mode obtains turn markers
  exclusively from the tokenizer chat template. Neither mode inserts BOS.

### Megatron-Bridge

The Chimera bridge validates and preserves the explicit phase, position type, maximum,
factor, original context, beta values, mscale values, theta, truncation behavior, and RMS
epsilon in both conversion directions.

## HF artifact verification

A final 128K export must contain equivalent values in `config.json`:

```json
{
  "context_phase": "128k",
  "position_embedding_type": "yarn",
  "max_position_embeddings": 131072,
  "original_max_position_embeddings": 8192,
  "rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 8192,
    "truncate": false
  }
}
```

The tokenizer, chat template, vocabulary, and model weight shapes do not change during
context extension.

## Validation gates

Do not declare a context phase complete based only on successful memory allocation or a
single needle test. Before advancing or publishing, verify:

- Short-context validation loss and benchmark quality recover after the transition.
- Long-document validation loss improves at the phase target.
- Evaluation covers 8K, intermediate lengths, and the target length.
- Retrieval tests vary both the target position and total prompt length.
- Real-document QA, summarization, or repository understanding works at long lengths.
- MCore and exported HF logits agree on the same short and long inputs within the expected
  numerical tolerance.
- Raw HF inference succeeds near the final context without configuration or cache errors.
- The exported HF config contains the exact final maximum, factor, and original maximum.

## Common mistakes

- Increasing only `--max-position-embeddings` and assuming the model learned long context.
- Applying the final YaRN factor during short training without intentionally accepting the
  changed short-context positional geometry.
- Resuming canonical training from the legacy CUDA-graph checkpoint that did not receive
  rotary inputs after graph capture.
- Changing `yarn_original_max_position_embeddings` during extension.
- Combining a factor and maximum that represent different target contexts.
- Publishing an intermediate checkpoint with final-context metadata.
- Reusing a stale `run_config.yaml` or data cache from another context phase.
- Using only unrelated short documents in the long-context corpus.
- Evaluating only whether a long forward pass runs instead of whether the model uses distant
  information correctly.
