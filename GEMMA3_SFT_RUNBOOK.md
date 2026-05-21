# Gemma3-1B SFT End-to-End Runbook

This guide provides the definitive steps to execute the Supervised Fine-Tuning (SFT) pipeline for Gemma3-1B from scratch on a remote GPU node.

## 📋 Prerequisites

Before starting, ensure you have the following:
1. **SSH Access**: Credentials for the target GPU node.
2. **GitHub Token**: A Personal Access Token (PAT) with repository read access (if the repo is private) to pull the `gemma3-1b` branch.
3. **HuggingFace Token**: A token with access to the `google/gemma-3-1b-pt` repository.

---

## 🚀 Step 1: Repository Setup

SSH into the node and navigate to the root directory. Clone the repository and switch to the active SFT branch:

```bash
cd /root
# If prompted, use your GitHub Token as the password
git clone -b gemma3-1b https://github.com/surya-vikram/Megatron-LM.git
cd Megatron-LM
```

---

## 🛠️ Step 2: Environment Initialization

Run the automated sanity check script. This will:
- Install `uv` (Fast Package Manager).
- Install Megatron-Core and dependencies.
- Install FlashAttention-3 (FA3) and patch the TransformerEngine backend.
- Verify GPU health.

```bash
bash sanity_check.sh
```

---

## 🌉 Step 3: Megatron-Bridge Installation

Install the Megatron-Bridge to enable conversion between HuggingFace and Megatron formats:

```bash
mkdir -p /home/jovyan
git clone https://github.com/surya-vikram/Megatron-Bridge.git /home/jovyan/Megatron-Bridge
cd /home/jovyan/Megatron-Bridge
git checkout main
pip install -e . --no-deps
```

---

## 🎯 Step 4: Execute SFT Pipeline

Run the master orchestration script. This script automates the entire flow:
1. **Import**: Converts HF model to Megatron-Core.
2. **Prepare**: Fetches Capybara dataset, injects "Gold" samples, and filters for 16k sequence length.
3. **Train**: Runs the optimized SFT loop with precision masking.
4. **Export**: Converts fine-tuned weights back to HuggingFace.
5. **Verify**: Validates memorization and chat template alignment.

```bash
cd /root/Megatron-LM
./examples/gemma3/run_1b_sft_pipeline.sh <YOUR_HF_TOKEN>
```

---

## 📊 Monitoring & Verification

- **GPU Utilization**: Run `nvidia-smi` in a separate terminal. Expect **95%+ utilization** due to our optimized dataloader configuration (`--num-workers 8` and hoisted token lookups).
- **Masking Logic**: The training uses the custom `gemma3` prompt format in `SFTTokenizer` which correctly masks the 3-token assistant header and shuts off loss exactly at `<end_of_turn>`.
- **Success Criteria**: The pipeline is successful if `verify_sft_results.py` (Step 5) reports a high memorization score and correct chat template alignment.

---

## 🛠️ Troubleshooting

- **Out of Memory**: Ensure you are on an H100/H200 node. The pipeline is configured for a 16k sequence length.
- **Tokenization Bottleneck**: If GPU utilization is low/sawtooth, check CPU load; the `--num-workers 8` flag should be sufficient for most H100 hosts.
