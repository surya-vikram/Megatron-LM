# Gemma 3 Production Training Runbook (4B & 12B)

This runbook describes the modular workflow for importing, training, and exporting Gemma 3 models using Megatron-LM.

## 1. Environment Setup
Bootstrap the node with all dependencies (FlashAttn, TE, Bridge) and persistent symlinks.
```bash
bash examples/gemma3/setup.sh
source /home/jovyan/load_env.sh
```

## 2. Import: HF VLM -> Megatron
Convert the original Google VLM into a Megatron-Core text backbone.
```bash
bash examples/gemma3/import.sh \
    --hf-model /path/to/gemma-3-12b-pt \
    --mcore-path /path/to/gemma-3-12b-pt-mcore
```

## 3. Preprocess: Prepare Training Data
Convert raw text to binary for CPT, or validate JSONL for SFT.

### For CPT (Raw Text to Binary)
```bash
bash examples/gemma3/preprocess.sh \
    --mode cpt \
    --input /path/to/my_data.txt \
    --output-prefix /path/to/my_data_bin \
    --hf-tokenizer /path/to/gemma-3-pt
```

### For SFT (Validate Instruction Data)
```bash
bash examples/gemma3/preprocess.sh \
    --mode sft \
    --input /path/to/my_instructions.jsonl
```

## 4. Train: CPT or SFT
Launch unified training with Precision-Aware Adam and dynamic model detection.

### Weights & Biases (WandB) Integration
WandB is pre-installed via `setup.sh` and is enabled by default in the training scripts. Megatron-LM logs key metrics (training loss, learning rate, GPU memory allocation, validation loss, throughput) automatically.

> [!NOTE]
> WandB logs are initialized only on the last rank (master process) to prevent duplicate runs and log clutter in multi-GPU distributed environments.

#### A. Authentication
Before launching training, you must authenticate your node with WandB. You can do this in two ways:
1. **Interactive Login**:
   ```bash
   wandb login
   ```
2. **Environment Variable** (recommended for automated scripts or background `nohup` jobs):
   ```bash
   export WANDB_API_KEY="your_wandb_api_key_here"
   ```

#### B. Script Parameters & Default Projects
The training script `train.sh` supports the following custom WandB flags:
* `--wandb-project`: Set the WandB project name.
  * **SFT Default**: `gemma3-medical-sft-reasoning`
  * **CPT Default**: `gemma3-medical-cpt-prod`
  * **Disable logging**: Pass `--wandb-project NONE` or `--wandb-project ""` to completely turn off WandB.
* `--wandb-exp-name`: Set a custom experiment run name. Defaults to `gemma3-${MODEL_SIZE}-${MODE}`.

#### C. Running Offline (No Internet Access)
If your training cluster does not have direct internet access:
1. Set the WandB mode to offline before training:
   ```bash
   export WANDB_MODE=offline
   ```
2. Launch training as usual. Logs will be saved locally inside the training checkpoint directory: `/path/to/checkpoints/wandb/`.
3. After training finishes, synchronize the offline logs to the cloud from a machine with internet access:
   ```bash
   wandb sync /path/to/checkpoints/wandb/offline-run-*
   ```

#### D. Controlling the Loaded Checkpoint Iteration
When initializing training (e.g., starting SFT from a CPT checkpoint, or resuming a paused run), Megatron-LM reads the `latest_checkpointed_iteration.txt` pointer file located in the root of the checkpoint load directory (the path specified by `--mcore-path`).

To target a specific iteration checkpoint (such as loading iteration `1000` of CPT instead of a newer short debug run):
1. Navigate to the checkpoint parent directory (e.g., `/home/jovyan/data/checkpoints/gemma3-4b-cpt`).
2. Overwrite the `latest_checkpointed_iteration.txt` file with your desired iteration number:
   ```bash
   echo 1000 > /home/jovyan/data/checkpoints/gemma3-4b-cpt/latest_checkpointed_iteration.txt
   ```
3. Launch training. Megatron-LM will read this file and load the corresponding subdirectory (e.g., `iter_0001000`).

---

### Continual Pre-training (CPT)
```bash
bash examples/gemma3/train.sh \
    --mode cpt \
    --model-size 12b \
    --hf-model /path/to/gemma-3-12b-pt \
    --mcore-path /path/to/gemma-3-12b-pt-mcore \
    --data-path /path/to/my_data_bin_text_document \
    --save-path /path/to/checkpoints \
    --wandb-project my-custom-cpt-project \
    --wandb-exp-name gemma3-12b-cpt-run1
```

### Instruction Tuning (SFT)
```bash
bash examples/gemma3/train.sh \
    --mode sft \
    --model-size 4b \
    --hf-model /path/to/gemma-3-4b-pt \
    --mcore-path /path/to/gemma-3-4b-pt-mcore \
    --data-path /path/to/my_instructions.jsonl \
    --save-path /path/to/checkpoints \
    --wandb-project my-custom-sft-project \
    --wandb-exp-name gemma3-4b-sft-run1
```

## 5. Export: Megatron -> HF
Convert the trained weights back to the HuggingFace format.
### Standalone Text Model
```bash
bash examples/gemma3/export.sh \
    --target text \
    --mcore-path /path/to/trained_mcore \
    --hf-reference /path/to/gemma-3-pt \
    --save-path /path/to/exported_hf
```

### Stitched Multimodal VLM
```bash
bash examples/gemma3/export.sh \
    --target vlm \
    --mcore-path /path/to/trained_mcore \
    --hf-reference /path/to/gemma-3-pt \
    --save-path /path/to/stitched_vlm
```

---
**Note for 12B:** 12B requires TP=2 or PP=2 for production runs to avoid OOM on single-GPU (even H200). Update `train.sh` distributed flags accordingly for multi-GPU nodes.
