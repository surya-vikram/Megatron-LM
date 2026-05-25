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

| Parameter | CLI Flag | Description |
| :--- | :--- | :--- |
| **Validation** | `--valid-data-path` | Path to a separate `.bin/.idx` (CPT) or `.jsonl` (SFT). |
| **Split** | `--split 99,1,0` | Allocates 99% Train, 1% Val, 0% Test (Recommended). |
| **Learning Rate** | `--lr` | Peak learning rate (e.g., `1e-5`). |
| **Min LR** | `--min-lr` | The floor for cosine decay (Default: 10% of peak). |
| **Warmup** | `--warmup-iters` | Steps to ramp up LR (Recommend 1-5% of total). |
| **Decay** | `--lr-decay-iters` | Steps to decay LR (Recommend matching total iters). |
| **Seq Len** | `--seq-len` | Context length. 8192 is optimized for H200 throughput. |

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

---
*Created by Gemini CLI - v1.0-cpt-sft-verified*
