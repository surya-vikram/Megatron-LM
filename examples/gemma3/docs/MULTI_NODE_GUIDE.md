# Gemma 3 Multi-Node Cluster Training Guide

This guide describes how to run Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), and Preference Optimization (SimPO) on multi-node GPU clusters (such as NVIDIA H200/H100/A100 clusters) using Megatron-LM.

---

## 1. Parallelism Topology & Memory Allocations

Before launching, you must map the model parameters across the total GPUs in the cluster.

### A. Core Parallelism Dimensions:
*   **TP (Tensor Parallelism):** Intrasheet tensor partitioning. Cuts layer weights across GPUs.
*   **PP (Pipeline Parallelism):** Layer-by-layer sequence pipeline. Divides model layers across GPU ranks.
*   **DP (Data Parallelism):** Batch replica groups. Total GPUs $\div (\text{TP} \times \text{PP})$.

### B. Standard Topologies per Model Size:
For single-node (8 GPUs) or multi-node (e.g., 5 nodes $\times$ 8 GPUs = 40 GPUs):

| Model Size | Recommended TP | Recommended PP | DP (Single-Node, 8 GPUs) | DP (5-Node Cluster, 40 GPUs) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1B** | `1` | `1` | `8` | `40` | High throughput. Single-GPU replica. |
| **4B** | `1` | `1` | `8` | `40` | Easily fits in a single H100/H200/A100. |
| **12B** | `2` | `1` | `4` | `20` | TP=2 required for optimal hidden-dim splitting. |

---

## 2. Launch Option A: Slurm Cluster Scheduling (Recommended)

The easiest way to orchestrate multi-node jobs is using our template Slurm script located at [examples/gemma3/submit_multinode.sbatch](file:///C:/Users/surya/Documents/antigravity/quirky-bell/Megatron-LM/examples/gemma3/submit_multinode.sbatch).

### Basic Command:
```bash
sbatch --nodes=5 --gres=gpu:8 \
    --export=ALL,MODE=sft,MODEL_SIZE=12b,DATA_PATH=/datasets/megadata/sft/mock_sft.jsonl,MCORE_PATH=/datasets/megadata/mcore_models/gemma-3-12b-pt-mcore,TOKENIZER_MODEL=/datasets/megadata/hf_models/gemma-3-12b-pt \
    examples/gemma3/submit_multinode.sbatch
```

### Targeting Specific Nodes (`--nodelist`):
To train on a specific set of nodes rather than arbitrary cluster allocations (e.g., training only on `node3` and `node5`), use Slurm's `--nodelist` (or `-w`) parameter. Set `--nodes` to the number of nodes specified:
```bash
sbatch --nodes=2 --nodelist=node3,node5 \
    --export=ALL,MODE=sft,MODEL_SIZE=12b,DATA_PATH=/datasets/megadata/sft/mock_sft.jsonl,MCORE_PATH=/datasets/megadata/mcore_models/gemma-3-12b-pt-mcore,TOKENIZER_MODEL=/datasets/megadata/hf_models/gemma-3-12b-pt \
    examples/gemma3/submit_multinode.sbatch
```
*Note: The script automatically handles this. The first node listed (`node3` in this case) becomes the master (`node-rank=0` & `MASTER_ADDR`), and subsequent nodes become workers.*

### Specific Mode Workflows:

#### Continual Pre-Training (CPT) on 5 Nodes:
```bash
sbatch --nodes=5 --gres=gpu:8 \
    --export=ALL,MODE=cpt,MODEL_SIZE=12b,DATA_PATH=/datasets/megadata/cpt/corpus_bin_text_document,MCORE_PATH=/datasets/megadata/mcore_models/gemma-3-12b-pt-mcore,TOKENIZER_MODEL=/datasets/megadata/hf_models/gemma-3-12b-pt \
    examples/gemma3/submit_multinode.sbatch
```

#### Supervised Fine-Tuning (SFT) on 3 Nodes:
```bash
sbatch --nodes=3 --gres=gpu:8 \
    --export=ALL,MODE=sft,MODEL_SIZE=4b,DATA_PATH=/datasets/megadata/sft/sft_dataset.jsonl,MCORE_PATH=/datasets/megadata/mcore_models/gemma-3-4b-pt-mcore,TOKENIZER_MODEL=/datasets/megadata/hf_models/gemma-3-4b-pt \
    examples/gemma3/submit_multinode.sbatch
```

#### Preference Optimization (SimPO) on 2 Nodes:
```bash
sbatch --nodes=2 --gres=gpu:8 \
    --export=ALL,MODE=simpo,MODEL_SIZE=4b,DATA_PATH=/datasets/megadata/preference/simpo_dataset.jsonl,MCORE_PATH=/datasets/megadata/mcore_models/gemma-3-4b-sft-mcore,TOKENIZER_MODEL=/datasets/megadata/hf_models/gemma-3-4b-pt \
    examples/gemma3/submit_multinode.sbatch
```

---

## 3. Launch Option B: Bare-Metal Manual Launching (No Slurm)

If you are running on custom virtual machines or bare-metal setups without a scheduler, you must execute the script manually on **each node**.

1.  **Select Node 0 as Master Node** and identify its internal IP address (e.g. `10.128.0.4`).
2.  **Ensure port `6000` is open** and not blocked by firewall rules (`ufw` or `iptables`) between nodes.
3.  **Run the script on all nodes** using their respective rank:

### Node 0 (Master):
```bash
./examples/gemma3/train.sh \
    --mode sft \
    --model-size 12b \
    --mcore-path /datasets/megadata/mcore_models/gemma-3-12b-pt-mcore \
    --tokenizer-model /datasets/megadata/hf_models/gemma-3-12b-pt \
    --data-path /datasets/megadata/sft/sft_dataset.jsonl \
    --nnodes 5 \
    --node-rank 0 \
    --master-addr 10.128.0.4 \
    --master-port 6000
```

### Node 1:
```bash
./examples/gemma3/train.sh \
    --mode sft \
    --model-size 12b \
    --mcore-path /datasets/megadata/mcore_models/gemma-3-12b-pt-mcore \
    --tokenizer-model /datasets/megadata/hf_models/gemma-3-12b-pt \
    --data-path /datasets/megadata/sft/sft_dataset.jsonl \
    --nnodes 5 \
    --node-rank 1 \
    --master-addr 10.128.0.4 \
    --master-port 6000
```
*(Repeat the command for Nodes 2, 3, and 4, incrementing `--node-rank` to `2`, `3`, and `4`).*

---

## 4. Multi-Node Networking & Troubleshooting

Multi-node training requires fast inter-GPU communication. The standard communication backend is NVIDIA NCCL.

### A. InfiniBand vs. RoCE/Ethernet
By default, the Slurm script enables InfiniBand (`NCCL_IB_DISABLE=0`). If your cluster does not have dedicated InfiniBand fabrics and uses high-speed Ethernet (or RoCE):

Add this configuration to your job environment variables:
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0   # Replace eth0 with your network card interface (e.g. bond0)
```

### B. Troubleshooting Initial Rendezvous Hangs
If your training script starts but hangs indefinitely before printing step logs:

1.  **Check Firewall Blockages:** Run `telnet <master_ip> 6000` from worker nodes to ensure connection to the master port is open.
2.  **NCCL Debug Logs:** Change debug level to capture transport information:
    ```bash
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
    ```
    This will output details to logs pointing exactly to the network interface causing the hang.
3.  **CUDA Memory Fragmentation:** If training fails on rank allocation with Out Of Memory (OOM) errors during startup, enable PyTorch allocator memory packing:
    ```bash
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ```
