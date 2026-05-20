#!/bin/bash
set -e
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH
python /home/jovyan/Megatron-Bridge/examples/conversion/hf_megatron_roundtrip.py \
    --hf-model-id /home/jovyan/models/gemma-3-1b-pt-hf \
    --output-dir /home/jovyan/models/gemma-3-1b-final-hf \
    --trust-remote-code
