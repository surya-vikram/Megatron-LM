# Chimera Context Extension

This document describes how to pretrain Chimera primarily at an 8K sequence length and
then extend it to 32K, 64K, or 128K. It separates the values that control actual training,
the declared model limit, and YaRN's positional-frequency transformation.

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

In the current Chimera Megatron implementation, an explicitly supplied YaRN factor and
`yarn_original_max_position_embeddings` determine the rotary frequencies. The maximum
position value is primarily a bound and checkpoint/export metadata. It is numerically
possible to declare the final maximum during the 8K phase while retaining factor `1`, but
phase-specific maximums are recommended because every checkpoint then advertises only a
context length on which it has actually been trained and validated.

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

The model does not necessarily need every row in the table. A 64K target can use
`8K -> 32K -> 64K`. A 128K target can use `8K -> 32K -> 128K`, adding a 64K phase if
the direct 32K-to-128K transition does not recover short-context quality or pass the
long-context validation gates.

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
iteration counters, optimizer state, RNG state, and scheduler. This avoids accidentally
restoring the base phase's completed WSD schedule. Preserve optimizer state only after a
separate transition test confirms that the checkpoint scheduler and changed parallelism
are handled correctly.

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

## Files to update

### Megatron-LM

`examples/chimera/pretrain_chimera.py`

- Set `yarn_rotary_scaling_factor` for the current phase.
- Keep `yarn_original_max_position_embeddings=8192`.
- Prefer making the phase factor an explicit training input instead of manually changing a
  global constant between jobs.

The phase training script:

- Set `--seq-length` and `--max-position-embeddings` from the phase table.
- Set MBS, GBS, CP, iteration count, and the extension LR schedule.
- Use a dedicated extension script instead of overwriting the production 8K recipe.

`examples/chimera/run_config.yaml`

- Set `seq_length` to the phase maximum.
- Set `yarn_rotary_scaling_factor` to the phase factor.
- Keep `yarn_original_max_position_embeddings: 8192`.
- Ensure the matching run config is stored beside the checkpoint before HF export.

`examples/chimera/sft.sh` and `examples/chimera/simpo.sh`

- After the final context extension, set `--max-position-embeddings` to the final supported
  context so post-training exports do not regress the model metadata.
- Their actual `SEQ_LENGTH` may remain shorter when the post-training data is short.

### Transformers

`src/transformers/models/chimera/configuration_chimera.py`

- Set the published default `max_position_embeddings` and YaRN factor to the final,
  validated context configuration.

`src/transformers/models/chimera/scripts/export_to_hf.py`

- Instantiate the final maximum, factor, and original maximum in the generated HF config.

These Transformers defaults should describe the final model, not an intermediate 8K or
32K checkpoint.

### Megatron-Bridge

No production mapping change is expected. The Chimera bridge already transfers maximum
position and YaRN fields. Update its Chimera test fixture when the published defaults move
from 32K to 64K or 128K.

## HF artifact verification

A final 128K export must contain equivalent values in `config.json`:

```json
{
  "max_position_embeddings": 131072,
  "original_max_position_embeddings": 8192,
  "rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 8192
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
- Changing `yarn_original_max_position_embeddings` during extension.
- Combining a factor and maximum that represent different target contexts.
- Publishing an intermediate checkpoint with final-context metadata.
- Reusing a stale `run_config.yaml` or data cache from another context phase.
- Using only unrelated short documents in the long-context corpus.
- Evaluating only whether a long forward pass runs instead of whether the model uses distant
  information correctly.
