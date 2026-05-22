#!/bin/bash
set -euo pipefail

# install_bridge.sh: Managed dependency installation for Megatron-Bridge

REPOS_DIR="/home/jovyan/repos"
BRIDGE_DIR="${REPOS_DIR}/Megatron-Bridge"

mkdir -p "${REPOS_DIR}"
cd "${REPOS_DIR}"

if [ ! -d "${BRIDGE_DIR}" ]; then
    echo "--- Cloning Megatron-Bridge --- "
    git clone https://github.com/surya-vikram/Megatron-Bridge.git "${BRIDGE_DIR}"
fi

cd "${BRIDGE_DIR}"
git checkout main
git pull --ff-only origin main

echo "--- Installing Bridge Dependencies --- "
python3 -m pip install "nvidia-resiliency-ext==0.6.0"
python3 -m pip install -e . --no-deps
echo "Megatron-Bridge installed successfully."
