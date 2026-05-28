# Gemma 3 Production Training Runbook

This runbook describes the end-to-end workflow for importing, training (CPT → SFT → SimPO), and exporting Gemma 3 models using Megatron-LM.

---

## 1. Environment Setup

You can choose between a standard bare-metal VM bootstrap or the highly recommended container-based setup for production/multi-node clusters.

### Option A: Self-Contained Container Setup (Recommended for Production & Slurm)

For a fully isolated, pre-compiled, and self-contained environment containing all dependencies (Megatron Core, FlashAttention-3, FlashInfer, Cut-Cross-Entropy, and Megatron-Bridge), use our optimized Docker workflow.

#### 1. Network MTU Pre-requisite (If experiencing SSL/Handshake Timeouts)
In virtualized cloud environments (overlay networks), large cryptographic packets during `git clone` or SSL handshakes can be dropped. To fix this, set your host network interface's MTU to `1400` before building:
```bash
sudo ip link set dev enp3s0 mtu 1400 # Replace enp3s0 with your primary interface
```

#### 2. Build the self-contained image
Run the build from the repository root using `--network=host` to inherit the MTU fix:
```bash
docker build --network=host -f docker/Dockerfile.gemma3 -t suryavikram6/megatron-gemma:latest .
```

#### 3. Run the container locally (with dynamic repository hot-reloading)
Keep the virtual environment outside `/workspace/repos` so that mounting local folders at runtime does not mask your compiled libraries. Any edits to Python files on the host are immediately reflected inside the container:
```bash
docker run --gpus all -it --rm \
  -v /path/to/dataset:/workspace/data \
  -v /path/to/checkpoints:/workspace/models \
  -v /path/to/logs:/workspace/logs \
  -v /path/to/repos:/workspace/repos \
  suryavikram6/megatron-gemma:latest
```

#### 4. Run Multi-Node Multi-GPU Training under Slurm
Slurm clusters typically use **Pyxis + Enroot** or **Apptainer/Singularity**. No Dockerfile modifications are needed—simply use the following Slurm launch commands:

* **Pyxis/Enroot**:
  ```bash
  srun --container-image="suryavikram6/megatron-gemma:latest" \
       --container-mounts="/path/to/dataset:/workspace/data,/path/to/checkpoints:/workspace/models,/path/to/logs:/workspace/logs" \
       bash -c "source /workspace/load_env.sh && python3 /workspace/repos/Megatron-LM/pretrain_gpt.py [YOUR_ARGS]"
  ```
* **Apptainer (Singularity)**:
  ```bash
  srun apptainer exec --nv \
    --bind /path/to/dataset:/workspace/data \
    --bind /path/to/checkpoints:/workspace/models \
    docker://suryavikram6/megatron-gemma:latest \
    bash -c "source /workspace/load_env.sh && python3 /workspace/repos/Megatron-LM/pretrain_gpt.py [YOUR_ARGS]"
  ```

---

### Option B: Bare-Metal VM Setup (Standard Bootstrap)

Bootstrap the node dynamically with all dependencies and persistent symlinks on a standard Ubuntu/JupyterHub cloud instance.
```bash
bash examples/gemma3/setup.sh
source /home/jovyan/load_env.sh
```

---

## 2. Import: HuggingFace → Megatron

Convert the original Google HF checkpoint into a Megatron-Core checkpoint.
```bash
bash examples/gemma3/import.sh \
    --hf-model /path/to/gemma-3-4b-pt \
    --mcore-path /path/to/gemma-3-4b-pt-mcore
```

---

## 3. Preprocess: Prepare Training Data

### CPT — Raw Text to Megatron Binary
```bash
bash examples/gemma3/preprocess.sh \
    --mode cpt \
    --input /path/to/corpus.txt \
    --output-prefix /path/to/corpus_bin \
    --hf-tokenizer /path/to/gemma-3-4b-pt
```
> **Note:** Pass `--data-path` to `train.sh` **without** the `.bin`/`.idx` extension.  
> Correct: `--data-path /path/to/corpus_bin_text_document`

### SFT / SimPO — Validate JSONL
```bash
bash examples/gemma3/preprocess.sh \
    --mode sft \
    --input /path/to/instructions.jsonl
```

**SFT JSONL format** — each line must have a `messages` key:
```json
{"messages": [
  {"role": "system",    "content": "You are a medical assistant."},
  {"role": "user",      "content": "What is hypertension?"},
  {"role": "assistant", "content": "Hypertension is..."}
]}
```

**SimPO JSONL format** — each line must have a `conversations` key with exactly 2 conversations (chosen + rejected):
```json
{"conversations": [
  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "chosen response"}],
  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "rejected response"}]
]}
```

---

## 4. Train

`train.sh` uses a **tiered argument system** — most things are automatic. You only need to supply what's in Tier 1.

```
TIER 1 — IDENTITY        Must pass. Controls everything.
TIER 2 — BUDGET          One arg per mode (token-budget / epochs / iters).
TIER 3 — LEARNING RATE   Auto-set per mode. Override with --lr.
TIER 4 — BATCH & SEQ     Auto per mode. Override with --global-batch-size / --seq-len.
TIER 5 — ALGO PARAMS     SimPO-specific knobs (beta, gamma, etc.).
TIER 6 — INFRA           Expert overrides (save paths, WandB, parallelism, etc.).
```

### Minimal Commands (copy-paste ready)

#### Continual Pre-Training (CPT)
Trains on 500M tokens by default. Adjust with `--token-budget`.
```bash
bash examples/gemma3/train.sh \
    --mode cpt \
    --model-size 4b \
    --mcore-path /home/jovyan/models/gemma-3-4b-pt-mcore \
    --data-path /home/jovyan/data/pubmed_train_text_document
```

#### Supervised Fine-Tuning (SFT)
Trains for 1 epoch by default. Adjust with `--epochs`. Packing is always on.
```bash
bash examples/gemma3/train.sh \
    --mode sft \
    --model-size 4b \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-cpt \
    --data-path /home/jovyan/data/sft_train.jsonl
```

#### Preference Optimization (SimPO)
Trains for 1 epoch by default. Adjust with `--epochs`. Packing is always on.
```bash
bash examples/gemma3/train.sh \
    --mode simpo \
    --model-size 4b \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-sft \
    --data-path /home/jovyan/data/dpo_mix.jsonl
```

### Adding Validation

Pass `--valid-data-path` to any mode. If omitted, validation is completely disabled.
```bash
# CPT with validation
bash examples/gemma3/train.sh \
    --mode cpt --model-size 4b \
    --mcore-path /home/jovyan/models/gemma-3-4b-pt-mcore \
    --data-path /home/jovyan/data/pubmed_train_text_document \
    --valid-data-path /home/jovyan/data/pubmed_val_text_document

# SFT with validation
bash examples/gemma3/train.sh \
    --mode sft --model-size 4b \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-cpt \
    --data-path /home/jovyan/data/sft_train.jsonl \
    --valid-data-path /home/jovyan/data/sft_val.jsonl
```

### Per-Mode Automatic Defaults

| Setting | CPT | SFT | SimPO |
|---|---|---|---|
| **Budget control** | `--token-budget 500M` | `--epochs 1.0` | `--epochs 1.0` |
| **GBS** | 64 | 32 | 32 |
| **LR** | 1e-5 | 5e-6 | 1e-6 |
| **Min LR** | 1e-6 | 5e-7 | 1e-7 |
| **LR warmup** | 2% of iters | 5% of iters | 5% of iters |
| **LR decay** | 90% of iters | 90% of iters | 90% of iters |
| **Sequence packing** | N/A (GPT native) | Always on | Always on |
| **No-validation split** | `100,0,0` | eval disabled | eval disabled |
| **Fused CCE loss** | ✅ | ✅ | ❌ (needs full logits) |

### Common Overrides & Expert Tuning (Tiers 2–4)

You can tune any aspect of your training run either by editing the `★ USER CONFIGURATION` block at the top of `train.sh` or by passing them as CLI overrides (CLI flags always win).

#### 1. Identity & Checkpoint Resumption
* **`--save-path`** (or `SAVE_PATH`): Specifies the directory to save training checkpoint outputs.
* **`--resume-from-checkpoint`** (or `RESUME_FROM_CHECKPOINT=true`): **Crucial for production.** When set to `true`, it tells Megatron-LM to load the full optimizer state and RNG seeds to seamlessly resume a crashed or preempted run from the exact iteration step it left off, rather than starting a clean fine-tune.

#### 2. Durations, Budgets & Evaluation frequency
* **`--iters`** (or `ITERS`): Hard step count override (skips token/epoch auto-calculations).
* **`--epochs`** (or `EPOCHS`): Controls SFT/SimPO duration (default: `1.0`).
* **`--token-budget`** (or `TOKEN_BUDGET`): Controls CPT duration (default: `500000000`).
* **`--save-interval`** (or `SAVE_INTERVAL`): Checkpoint frequency in steps.
* **`--eval-interval`** (or `EVAL_INTERVAL`): Validation frequency in steps.
* **`--eval-iters`** (or `EVAL_ITERS`): Number of validation steps to run to compute validation loss (default: `2`).

#### 3. Optimizer & Hyperparameter Tuning
* **`--lr`** (or `LR`): Peak learning rate.
* **`--min-lr`** (or `MIN_LR`): Floor to decay the learning rate to.
* **`--lr-decay-style`** (or `LR_DECAY_STYLE`): Decay schedule shape (e.g. `cosine`, `linear`, `constant`).
* **`--weight-decay`** (or `WEIGHT_DECAY`): Optimizer L2 regularization coefficient (default: `0.1`).
* **`--clip-grad`** (or `CLIP_GRAD`): Gradient norm clipping limit to prevent exploding gradients (default: `1.0`).
* **`--adam-beta1`** & **`--adam-beta2`**: Adam optimizer beta moments (default: `0.9` / `0.95`).

#### 4. Hardware Scaling & VRAM footprint Optimization
* **`--global-batch-size`** (or `GBS`): Global batch size.
* **`--micro-batch-size`** (or `MBS`): Micro-batch size (shrink this to `1` if you hit OOMs).
* **`--seq-len`** (or `SEQ_LEN`): Model sequence/context length (default: `8192`).
* **`--recompute-granularity`** (or `RECOMPUTE_GRANULARITY`): Memory saving activation recomputation settings (`auto | none | selective | full`). Set to `full` to fit larger batches at long sequence lengths.
* **`--num-workers`** (or `NUM_WORKERS`): Number of CPU dataloader threads per GPU (default: `4`). Adjust if dataloading is a bottleneck or memory is tight.

### SimPO Algorithm Knobs (Tier 5)

```bash
--simpo-beta 2.0        # reward scaling (default: 2.0)
--simpo-gamma 0.5       # target margin (default: 0.5)
--simpo-loss-type sigmoid  # loss function (default: sigmoid)
--simpo-sft-weight 0.1  # add SFT regularization (default: 0.0)
```

---

## 5. Parallelism

The script automatically configures parallelism based on model size and available GPUs. You rarely need to change anything here.

### Automatic Defaults

| Model | TP | PP | DP (example, 4 GPUs) |
|---|---|---|---|
| 1b | 1 | 1 | 4 |
| 4b | 1 | 1 | 4 |
| 12b | 2 | 1 | 2 |

- **TP (Tensor Parallel):** splits model weight matrices across GPUs within a node. Higher TP = less memory per GPU, but more inter-GPU communication.
- **PP (Pipeline Parallel):** splits layers across GPUs. Always `PP=1` by default (not recommended to change for single-node fine-tuning).
- **DP (Data Parallel):** computed automatically as `NUM_GPUS / (TP × PP)`. Each DP replica processes a different micro-batch.
- **SP (Sequence Parallel):** automatically enabled when `TP > 1`. Distributes layer-norm and dropout activations across the TP group to save memory.

> `NUM_GPUS` is auto-detected via `nvidia-smi`. The script will error out if `TP × PP > NUM_GPUS`.

### Limiting GPU Visibility

If you're on a shared node and want to use only some GPUs, restrict visibility **before** launching:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3   # use 4 of 8 GPUs
bash examples/gemma3/train.sh --mode sft ...
```
The script reads `NUM_GPUS` from `nvidia-smi`, so it will see exactly the GPUs you expose.

### Manual TP Override

```bash
# Force TP=4 on a 4-GPU node for a 12b model to maximize memory savings
bash examples/gemma3/train.sh --mode cpt --model-size 12b \
    --tp-size 4 \
    --mcore-path ... --data-path ...
```

### Multi-Node Training

All nodes must share the same model/data via NFS or equivalent. Launch on each node simultaneously:

**Node 0 (master):**
```bash
bash examples/gemma3/train.sh \
    --mode cpt --model-size 4b \
    --mcore-path /path/mcore \
    --data-path /path/data \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr 10.0.0.1 \
    --master-port 6789
```

**Node 1 (worker):**
```bash
bash examples/gemma3/train.sh \
    --mode cpt --model-size 4b \
    --mcore-path /path/mcore \
    --data-path /path/data \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr 10.0.0.1 \
    --master-port 6789
```

> Global batch size remains constant across nodes — Megatron distributes the work automatically across `NNODES × NUM_GPUS` total GPUs. Effective DP = `(NNODES × NUM_GPUS) / (TP × PP)`.

### Parallelism Quick Reference

| Flag | Default | What it does |
|---|---|---|
| `--tp-size` | auto (1 or 2) | Tensor parallel degree |
| `--nnodes` | 1 | Number of nodes in cluster |
| `--node-rank` | 0 | Rank of this node (0 = master) |
| `--master-addr` | `localhost` | IP of the master node |
| `--master-port` | 6789 | Port for distributed rendezvous |

---

## 6. WandB Integration

WandB is enabled by default with a per-mode project name.

| Mode | Default Project |
|---|---|
| CPT | `gemma3-medical-cpt` |
| SFT | `gemma3-medical-sft` |
| SimPO | `gemma3-medical-simpo` |

```bash
# Authenticate once
export WANDB_API_KEY="your_key_here"

# Override project and run name
--wandb-project my-project --wandb-exp-name my-run-name

# Disable WandB entirely
--wandb-project NONE
```

**Offline mode** (no internet):
```bash
export WANDB_MODE=offline
# After training, sync from a machine with internet:
wandb sync /path/to/checkpoints/wandb/offline-run-*
```

---

## 7. Loading a Specific Checkpoint Iteration

Megatron reads `latest_checkpointed_iteration.txt` from the `--mcore-path` directory to decide which checkpoint to load. To target a specific iteration:

```bash
echo 1000 > /home/jovyan/data/checkpoints/gemma3-4b-cpt/latest_checkpointed_iteration.txt
```

Then launch training normally. Megatron will load `iter_0001000/`.

---

## 8. Export: Megatron → HuggingFace

Convert trained weights back to HF format for inference or deployment.

### Text Model
```bash
bash examples/gemma3/export.sh \
    --target text \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-sft \
    --hf-reference /home/jovyan/models/gemma-3-4b-pt \
    --save-path /home/jovyan/models/gemma3-4b-sft-hf
```

### Stitched VLM (Multimodal)
```bash
bash examples/gemma3/export.sh \
    --target vlm \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-cpt \
    --hf-reference /home/jovyan/models/gemma-3-4b-pt \
    --save-path /home/jovyan/models/gemma3-4b-cpt-vlm
```

---

## 9. Full Pipeline (CPT → SFT → SimPO)

```bash
# Step 1: CPT on domain corpus
bash examples/gemma3/train.sh --mode cpt --model-size 4b \
    --mcore-path /home/jovyan/models/gemma-3-4b-pt-mcore \
    --data-path /home/jovyan/data/pubmed_train_text_document

# Step 2: SFT on instruction data
bash examples/gemma3/train.sh --mode sft --model-size 4b \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-cpt \
    --data-path /home/jovyan/data/sft_train.jsonl

# Step 3: SimPO on preference pairs
bash examples/gemma3/train.sh --mode simpo --model-size 4b \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-sft \
    --data-path /home/jovyan/data/dpo_mix.jsonl

# Step 4: Export final model
bash examples/gemma3/export.sh --target text \
    --mcore-path /home/jovyan/data/checkpoints/gemma3-4b-simpo \
    --hf-reference /home/jovyan/models/gemma-3-4b-pt \
    --save-path /home/jovyan/models/gemma3-4b-final-hf
```

---

> **12B note:** 12B requires TP=2 by default (set automatically). On shared nodes, restrict GPU visibility first: `export CUDA_VISIBLE_DEVICES=0,1,2,3`
