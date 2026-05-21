# Gemma3-1B Automated Verification Runbook

This runbook describes how to execute the end-to-end verification for Gemma3-1B.

## 1. Environment Setup
```bash
git clone -b gemma3-1b https://github.com/surya-vikram/Megatron-LM.git
cd Megatron-LM
# Note: Ensure you are on the verified commit 043ae492f or later
bash sanity_check.sh
```

## 2. Megatron-Bridge Setup
Ensure the Megatron-Bridge is cloned and in the correct path:
```bash
mkdir -p /home/jovyan
git clone https://github.com/surya-vikram/Megatron-Bridge.git /home/jovyan/Megatron-Bridge
cd /home/jovyan/Megatron-Bridge
git checkout main
pip install -e . --no-deps
```

## 3. Execute Pipeline
Return to the Megatron-LM directory and run the automated flow. You must provide a HuggingFace token with access to the Gemma3 repository.

```bash
cd /root/Megatron-LM
./examples/gemma3/run_1b_pipeline.sh <YOUR_HF_TOKEN>
```

## 4. Success Criteria
The pipeline is successful if the final output prints:
`PIPELINE STATUS: [SUCCESS]`
`✓ Numerical Check: Global Max Difference = 9.5367431640625e-07`
