#!/bin/bash
set -e
BASE_DIR="/home/jovyan"
cd $BASE_DIR
if [ ! -d "Megatron-Bridge" ]; then
    git clone https://github.com/surya-vikram/Megatron-Bridge.git Megatron-Bridge
fi
cd Megatron-Bridge
git checkout main
git pull origin main

# Install globally to leverage the container's optimized PyTorch
pip install -e .
echo "Bridge dependencies installed globally."
