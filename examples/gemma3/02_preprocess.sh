#!/bin/bash
set -e

# Ensure data directory exists
mkdir -p /home/jovyan/data

# If medical data is not present, create a sample for the pipeline to work
if [ ! -f "/home/jovyan/data/medical_data.jsonl" ]; then
    echo "Creating sample medical data..."
    cat <<INTERNAL_EOF > /home/jovyan/data/medical_sample.txt
Patient presents with acute onset of shortness of breath and chest pain.
The patient has a history of hypertension and hyperlipidemia.
Physical examination reveals tachypnea and tachycardia.
INTERNAL_EOF
    python3 -c "import json; [print(json.dumps({\"text\": line.strip()})) for line in open(\"/home/jovyan/data/medical_sample.txt\", \"r\") if line.strip()]" > /home/jovyan/data/medical_data.jsonl
fi

source /home/jovyan/Megatron-Bridge-Surya/.venv/bin/activate
python /root/Megatron-LM/tools/preprocess_data.py     --input /home/jovyan/data/medical_data.jsonl     --output-prefix /home/jovyan/data/gemma_medical_data     --tokenizer-type HuggingFaceTokenizer     --tokenizer-model /home/jovyan/models/gemma-3-1b-pt-hf     --append-eod     --workers 1
