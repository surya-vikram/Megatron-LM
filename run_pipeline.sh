#!/bin/bash
# run_pipeline.sh: Orchestrates the full Gemma-3 1B training pipeline.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "--- Starting End-to-End Pipeline ---"

# 1. Base Dependency Setup (Assume already in repo)
cd "${SCRIPT_DIR}"
bash sanity_check.sh
bash install_bridge.sh

# 2. Gemma-3 1B Modular Pipeline
cd "${SCRIPT_DIR}/examples/gemma3"

# Run modular pipeline steps sequentially
bash 01_convert.sh
bash 02_preprocess.sh
bash 03_train.sh
bash 04_export.sh
bash 05_evaluate.sh

echo "--- Pipeline Execution Complete ---"
