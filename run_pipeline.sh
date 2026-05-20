#!/bin/bash
# run_pipeline.sh: Orchestrates the full Gemma-3 1B training pipeline.

set -e

echo "--- Starting End-to-End Pipeline ---"

# 1. Clean Environment and Clone Repository
cd /root
rm -rf Megatron-LM
git clone https://github.com/surya-vikram/Megatron-LM.git Megatron-LM
cd Megatron-LM
git checkout gemma3-1b

# 2. Base Dependency Setup
bash sanity_check.sh
bash install_bridge.sh

# 3. Gemma-3 1B Modular Pipeline
cd examples/gemma3/

# Run modular pipeline steps sequentially
bash 01_convert.sh
bash 02_preprocess.sh
bash 03_train.sh
bash 04_export.sh
bash 05_evaluate.sh

echo "--- Pipeline Execution Complete ---"
