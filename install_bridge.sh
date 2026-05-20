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
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
fi
echo "Bridge dependencies installed globally."
