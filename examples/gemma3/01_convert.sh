#!/bin/bash
set -e
source /home/jovyan/Megatron-Bridge/.venv/bin/activate
export HF_MODEL_PATH="/home/jovyan/models/gemma-3-1b-pt-hf"
export ROUNDTRIP_PATH="/home/jovyan/models/gemma-3-1b-roundtrip-test"
if [ ! -d "$HF_MODEL_PATH" ]; then
    hf download google/gemma-3-1b-pt --local-dir $HF_MODEL_PATH
fi
python /home/jovyan/Megatron-Bridge/examples/conversion/hf_megatron_roundtrip.py     --hf-model-id $HF_MODEL_PATH     --output-dir $ROUNDTRIP_PATH     --trust-remote-code
