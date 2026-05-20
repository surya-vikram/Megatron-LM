#!/bin/bash
set -e
source /home/jovyan/Megatron-Bridge-Surya/.venv/bin/activate
python /home/jovyan/Megatron-Bridge-Surya/examples/conversion/hf_megatron_roundtrip.py     --hf-model-id /home/jovyan/models/gemma-3-1b-pt-hf     --output-dir /home/jovyan/models/gemma-3-1b-final-hf     --trust-remote-code
