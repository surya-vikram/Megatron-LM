#!/bin/bash
set -euo pipefail

BASE_DIR="/home/jovyan"
BRIDGE_DIR="${BASE_DIR}/Megatron-Bridge"

cd "${BASE_DIR}"
if [ ! -d "${BRIDGE_DIR}" ]; then
    git clone https://github.com/surya-vikram/Megatron-Bridge.git "${BRIDGE_DIR}"
fi

cd "${BRIDGE_DIR}"
git checkout main
git pull --ff-only origin main

python3 -m pip install "nvidia-resiliency-ext==0.6.0"
python3 -m pip install -e . --no-deps
echo "Bridge dependencies installed globally."
