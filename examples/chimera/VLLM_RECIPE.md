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

The checkpoint does not need to be re-exported. The compatibility change does
not alter configuration fields, tensor shapes, or checkpoint key names. Stock
Transformers 5.9.0 does not register Chimera, so expose the complete fork via
`PYTHONPATH`. Do not copy individual files into the container's installed
Transformers package, and do not modify vLLM itself.

## Start The Server

Set host paths to the exported checkpoint and Transformers checkout:

```bash
export MODEL_DIR=/path/to/hf_sft_overfit
export TRANSFORMERS_DIR=/path/to/transformers

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
test -f "$TRANSFORMERS_DIR/src/transformers/models/chimera/modeling_chimera.py"
```

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

- Complete the 32k context-extension training stage from the validated 8k
  checkpoint.
- Verify the exported YaRN configuration retains an 8k original context, a
  32k maximum context, and factor 4.
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
