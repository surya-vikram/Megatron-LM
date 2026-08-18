# Chimera vLLM 0.22 Recipe

This recipe serves a complete Chimera HF checkpoint on one GPU with the
`vllm/vllm-openai:v0.22.0` container. The validated baseline is an H100 80GB,
BF16 weights, tensor parallel size 1, data parallel size 1, and an 8k maximum
context.

## Requirements

The Docker host must have:

- One CUDA-visible GPU with enough memory for the model and KV cache.
- A complete exported Chimera HF checkpoint, including all weight shards,
  `config.json`, tokenizer files, generation config, and chat template.
- The Chimera Transformers fork containing `ChimeraExperts`, the vLLM-compatible
  expert container.
- The `vllm/vllm-openai:v0.22.0` image.

The checkpoint does not need to be re-exported. Every checkpoint contains the
frozen router correction-bias tensors. The `load_with_bias` config field only
chooses whether routing applies those tensors; changing it must not alter any
tensor shape, key, or shard hash. Stock
Transformers 5.9.0 does not register Chimera, so expose the complete fork via
`PYTHONPATH`. Do not copy individual files into the container's installed
Transformers package, and do not modify vLLM itself.

## Start The Server

Set host paths to the exported checkpoint and Transformers checkout:

```bash
export MODEL_DIR=/path/to/hf_sft_overfit
export TRANSFORMERS_DIR=/path/to/transformers
export MEGATRON_LM=/path/to/Megatron-LM

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
test -f "$TRANSFORMERS_DIR/src/transformers/models/chimera/modeling_chimera.py"

python3 "$MEGATRON_LM/examples/chimera/architecture_contract.py" \
  validate-hf "$MODEL_DIR" --profile full --weights
```

The validation requires QK norm, 32×2048 routed experts, no shared expert,
8K/factor-1 YaRN, every expected model key, and every router bias. To test the
bias-bypassed mode with byte-identical weights, create a metadata-only variant:

```bash
export MODEL_DIR_NO_BIAS=/path/to/hf_sft_overfit_no_bias
test ! -e "$MODEL_DIR_NO_BIAS"
cp -al "$MODEL_DIR" "$MODEL_DIR_NO_BIAS"
python3 "$MEGATRON_LM/examples/chimera/architecture_contract.py" \
  set-load-with-bias "$MODEL_DIR_NO_BIAS" false
python3 "$MEGATRON_LM/examples/chimera/architecture_contract.py" \
  validate-hf "$MODEL_DIR_NO_BIAS" --profile full --weights
```

Use `MODEL_DIR` for `load_with_bias=true` and `MODEL_DIR_NO_BIAS` for false.
The hard-linked safetensors shards prove both modes use the same weights.

Start the container:

```bash
docker run -d \
  --name chimera-vllm \
  --gpus '"device=0"' \
  --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$MODEL_DIR:/models/chimera:ro" \
  -v "$TRANSFORMERS_DIR:/workspace/repos/transformers:ro" \
  -e PYTHONPATH=/workspace/repos/transformers/src \
  vllm/vllm-openai:v0.22.0 \
  --model /models/chimera \
  --served-model-name chimera \
  --model-impl transformers \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
```

Expert parallelism is intentionally disabled for this one-GPU configuration.
`--enforce-eager` makes the first correctness check easier to diagnose by
disabling `torch.compile` and CUDA graphs.

Run the server and the checks below once per model directory. Loader logs must
contain no missing or unexpected keys. For identical prompts, temperature 0,
and token limits, compare the exact generated token IDs with Transformers run
using `infer.py --expect-load-with-bias true` or `false`. Require
Transformers-vLLM equality within each mode; enabled and disabled modes are
allowed to differ from each other because they intentionally route differently.

## Verify Startup

Follow startup logs:

```bash
docker logs -f chimera-vllm
```

A successful load reports:

```text
Resolved architecture: TransformersMoEForCausalLM
Using Transformers modeling backend.
Using FlashInfer CUTLASS Unquantized MoE backend
Using FLASH_ATTN attention backend
```

Check the API:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

## Verify Chat Inference

For normal chat serving, stop on both the pretraining EOS token and chat turn
terminator. vLLM consumes the stop token, so the returned text does not include
`<end_of_turn>`:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chimera",
    "messages": [
      {"role": "system", "content": "Answer clearly and briefly."},
      {"role": "user", "content": "What was restored?"}
    ],
    "temperature": 0,
    "max_tokens": 16,
    "stop_token_ids": [1, 3]
  }'
```

The SFT overfit checkpoint used for validation returns:

```text
An old coastal map was restored.
```

To inspect the generated special token, omit `stop_token_ids`, disable special
token removal, and bound generation to the known completion length:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chimera",
    "messages": [
      {"role": "system", "content": "Answer clearly and briefly."},
      {"role": "user", "content": "What was restored?"}
    ],
    "temperature": 0,
    "max_tokens": 8,
    "skip_special_tokens": false
  }'
```

The raw response content is:

```text
An old coastal map was restored.<end_of_turn>
```

## Operate The Server

The API is bound to host loopback. Access it from another machine with an SSH
tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 user@vllm-host
```

Stop, start, or remove the container with:

```bash
docker stop chimera-vllm
docker start chimera-vllm
docker rm -f chimera-vllm
```

## TODO

### Compiled Execution

- Remove `--enforce-eager` after the eager correctness baseline passes.
- Repeat exact greedy-output checks with `torch.compile` and CUDA graphs.
- Measure first-request latency separately from warmed-up latency.

### 32k Context

- This work is deferred. The active configuration supports only an 8k maximum
  and original context with YaRN factor 1.
- In a future context-extension change, train from the validated 8k checkpoint
  before emitting a 32k maximum and factor 4.
- Compare HF and vLLM greedy output at short context before long-context tests.
- Exercise prompt lengths near 8k, 16k, and 32k and check generation and KV
  cache behavior.
- Change `--max-model-len` to `32768` only after those checks pass.

### Tensor Parallel Greater Than One

- Add and test Chimera `base_model_tp_plan` mappings in the Transformers config.
- Verify TP2 greedy-token parity against TP1 before performance testing.
- Test TP2 with expert parallel enabled on two H200 GPUs.
- Benchmark TP2 against TP1/DP2 using identical prompts and concurrency.
- Adopt TP greater than one only when memory requirements or measured
  throughput justify it.
