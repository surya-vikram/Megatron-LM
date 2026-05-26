# Gemma 3 Air-Gapped Operation & Experimentation Guide

This guide provides the essential technical knowledge required to operate the Gemma 3 training and conversion pipeline on isolated high-performance nodes (e.g., 4xH200 systems).

---

## 1. Container Operations (Run & Mount)
To launch the environment, use the following Docker command. Ensure you mount your physical drive containing the weights and data to the persistent `/home/jovyan` pillar.

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=64g \
  -v /mnt/drive/neuralix/jovyan:/home/jovyan \
  -it gemma3-megatron-airgap:latest
```

*   **CRITICAL DETAIL:** Always use `--shm-size=64g`. High-speed GPU-to-GPU communication (NCCL) requires significant shared memory. Without this, the training process may hang indefinitely or crash during the first iteration.

## 2. GPU Monitoring (Indices & Stats)
Before starting a run, verify the status and indices of your hardware.
*   **Real-time Stats:** Run `watch -n 1 nvidia-smi` to monitor power consumption, temperature, and VRAM utilization.
*   **Index Selection:** Identify the GPU indices (0, 1, 2, 3...) in the leftmost column of the `nvidia-smi` table. You will use these indices to isolate workloads if the node is shared.

## 3. Limiting GPU Visibility
If you are on a multi-GPU node (e.g., 8 GPUs) but only want to use a subset (e.g., 4 GPUs), isolate them using the environment variable **before** launching `train.sh`.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash examples/gemma3/train.sh ...
```
*   **Effect:** The script will automatically detect `NUM_GPUS=4` and configure the distributed backend to use only those devices.

## 4. Parallelism Logic (TP, DP, PP, SP)
The pipeline automatically scales based on the `--model-size` flag, but provides manual overrides for experimentation.

*   **Automatic Setup:**
    *   **1b/4b:** Defaults to `TP=1`, `PP=1`.
    *   **12b:** Defaults to `TP=2`, `PP=1`.
*   **DP (Data Parallel):** Calculated automatically as `NUM_GPUS / (TP * PP)`.
    *   *Example:* 4 GPUs, 12b model (`TP=2`, `PP=1`) results in `DP=2`.
*   **Manual Override:** Use the `--tp-size` flag to force a specific Tensor Parallelism.
    *   *Example:* To maximize VRAM savings for a 12b model on 4 GPUs: `--tp-size 4`.
*   **Sequence Parallel (SP):** Automatically enabled whenever `TP > 1` to distribute activations across the TP group.

## 5. Hyperparameter Sliders (Experimentation)
Adjust these flags in `train.sh` to tune your training runs:

| Tier | Parameter | CLI Flag | Description |
| :--- | :--- | :--- | :--- |
| 2 | **Budget (CPT)** | `--token-budget` | Tokens to train on (default: 500M). |
| 2 | **Budget (SFT/SimPO)** | `--epochs` | Epochs to train for (default: 1.0). |
| 2 | **Hard Step Override** | `--iters` | Explicit step count — skips all budget auto-calc. |
| 3 | **Learning Rate** | `--lr` | Peak LR. Auto: CPT=1e-5, SFT=5e-6, SimPO=1e-6. |
| 3 | **Min LR** | `--min-lr` | Floor for cosine decay (default: `lr × 0.1`). |
| 3 | **LR Warmup** | `--warmup-prct` | % of iters for warmup. Auto: CPT=2%, SFT/SimPO=5%. |
| 4 | **Batch Size** | `--global-batch-size` | Auto: CPT=64, SFT/SimPO=32. |
| 4 | **Seq Length** | `--seq-len` | Context length (default: 8192). |
| 6 | **Validation** | `--valid-data-path` | Separate val file. If omitted → no validation at all. |

## 6. Automated vs. Manual (The "Must Changes")
While much is automated, certain performance and state configurations require manual intervention:

*   **VRAM Management (MBS):** This cannot be fully automated. If you encounter an **Out of Memory (OOM)** error, manually decrease `--micro-batch-size` (e.g., from 4 to 2, or 1).
*   **Data Path Suffixes (CPT):** Megatron-LM requires files to end in `_text_document.bin`. When passing `--data-path` to the script, **omit the extension**.
    *   *Correct:* `--data-path /home/jovyan/data/my_file`
*   **Checkpoint Resuming:** To resume from a specific iteration, you must manually edit the text inside:
    `/home/jovyan/data/checkpoints/latest_checkpointed_iteration.txt`

## 7. Multi-Node Operation (Cluster Scaling)
Megatron-LM is designed for massive scaling. If you are deploying the container across multiple physical nodes (e.g., two 4xH200 nodes), you must coordinate them using the distributed CLI flags.

*   **Prerequisites:**
    *   All nodes must have the exact same `/home/jovyan/models` and `/home/jovyan/data` mounted (e.g., via NFS).
    *   Nodes must be able to ping each other over the local high-speed network.

**Example: 2 Nodes, 4 GPUs each**

*   **Node 0 (Master Node):**
    ```bash
    bash examples/gemma3/train.sh \
      --nnodes 2 \
      --node-rank 0 \
      --master-addr 10.0.0.1 \
      --master-port 6789 \
      [... other args]
    ```
*   **Node 1 (Worker Node):**
    ```bash
    bash examples/gemma3/train.sh \
      --nnodes 2 \
      --node-rank 1 \
      --master-addr 10.0.0.1 \
      --master-port 6789 \
      [... other args]
    ```
*   *Note:* The total Global Batch Size (`GBS`) remains constant; Megatron will automatically distribute the work across the `NNODES * NUM_GPUS` total GPUs.

## 8. Air-Gap Environment Variables
The following flags are **pre-baked** into the image but should be verified if you are running in a custom shell:
*   `export WANDB_MODE=offline`
*   `export TRANSFORMERS_OFFLINE=1`
*   `export HF_DATASETS_OFFLINE=1`

## 9. Environment Setup & PYTHONPATH (Manual Host Configuration)
If you are running outside the pre-baked Docker container (e.g., directly on a bare-metal server or VM environment), you must configure your shell environment manually. The container automatically has these environment values pre-set, but users running custom hostVMs must manually execute these configurations (which is what `source load_env.sh` typically performs):

1. **Activate your Python Virtual Environment:** Activate the environment containing PyTorch, Megatron dependencies, and cut-cross-entropy:
   ```bash
   source /home/jovyan/venv/bin/activate
   ```
2. **Configure Python Path:** You must place `Megatron-LM` and `Megatron-Bridge` (if utilizing Megatron-Bridge to implement Gemma 3 models) in your `PYTHONPATH` so that Python can locate all core modules:
   ```bash
   export PYTHONPATH="/home/jovyan/repos/Megatron-LM:/home/jovyan/repos/Megatron-Bridge/src:$PYTHONPATH"
   ```
3. **Configure Hugging Face Cache:** Force the Hugging Face cache directory to point to your persistent mounted disk to prevent it from consuming root VRAM disk space or redownloading model files repeatedly:
   ```bash
   export HF_HOME=/home/jovyan/models/.cache
   ```

---
*Updated: CPT/SFT/SimPO pipeline — tiered train.sh with CCE enabled by default, 100% accurate padding stats, and host environment guides.*
