#!/bin/bash
set -euo pipefail

HF_REFERENCE="/datasets/megadata/hf_models/chimera-10b"
MCORE_PATH="/datasets/megadata/chimera_runs/overfit/checkpoints"
HF_PATH="/datasets/megadata/hf_exports/chimera-overfit-hf"
BRIDGE_PATH="/workspace/repos/Megatron-Bridge"
PYTHON_BIN=""
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_CONFIG_SOURCE="$SCRIPT_DIR/run_config.yaml"

usage() {
    cat <<'USAGE'
Usage: bash examples/chimera/export.sh [options]

Options:
  --hf-reference PATH   Original HF Chimera model directory used as config/tokenizer reference.
  --mcore-path PATH     Trained Megatron-Core checkpoint root or iteration directory.
  --hf-path PATH        Output HF model directory.
  --bridge-path PATH    Megatron-Bridge repository path.
  --run-config PATH     Bridge run_config.yaml to install if checkpoint does not have one.
  --python PATH         Python executable.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-reference) HF_REFERENCE="$2"; shift 2 ;;
        --mcore-path) MCORE_PATH="$2"; shift 2 ;;
        --hf-path) HF_PATH="$2"; shift 2 ;;
        --bridge-path) BRIDGE_PATH="$2"; shift 2 ;;
        --run-config) RUN_CONFIG_SOURCE="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /workspace/venv/bin/python ]]; then
        PYTHON_BIN="/workspace/venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

[[ -d "$HF_REFERENCE" ]] || { echo "Missing HF reference: $HF_REFERENCE"; exit 1; }
[[ -d "$MCORE_PATH" ]] || { echo "Missing MCore checkpoint: $MCORE_PATH"; exit 1; }
[[ -d "$BRIDGE_PATH" ]] || { echo "Missing Megatron-Bridge repo: $BRIDGE_PATH"; exit 1; }

CHECKPOINT_ROOT="$MCORE_PATH"
CHECKPOINT_ITER=""
if [[ "$(basename "$MCORE_PATH")" == iter_* ]]; then
    CHECKPOINT_ROOT=$(dirname "$MCORE_PATH")
    CHECKPOINT_ITER="$MCORE_PATH"
else
    CHECKPOINT_ITER=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'iter_*' | sort -V | tail -n 1)
fi

if [[ ! -f "$CHECKPOINT_ROOT/run_config.yaml" ]]; then
    [[ -f "$RUN_CONFIG_SOURCE" ]] || {
        echo "Missing Bridge run config: $RUN_CONFIG_SOURCE" >&2
        echo "Pass --run-config PATH or keep examples/chimera/run_config.yaml available." >&2
        exit 1
    }
    cp "$RUN_CONFIG_SOURCE" "$CHECKPOINT_ROOT/run_config.yaml"
fi
echo "Using Bridge run config: $CHECKPOINT_ROOT/run_config.yaml"

if [[ -n "$CHECKPOINT_ITER" && ! -f "$CHECKPOINT_ITER/run_config.yaml" ]]; then
    cp "$CHECKPOINT_ROOT/run_config.yaml" "$CHECKPOINT_ITER/run_config.yaml"
    echo "Using Bridge run config: $CHECKPOINT_ITER/run_config.yaml"
fi

mkdir -p "$(dirname "$HF_PATH")"

export PYTHONPATH="$BRIDGE_PATH/src:${PYTHONPATH:-}"

"$PYTHON_BIN" "$BRIDGE_PATH/examples/conversion/convert_checkpoints.py" export \
    --hf-model "$HF_REFERENCE" \
    --megatron-path "$MCORE_PATH" \
    --hf-path "$HF_PATH" \
    --trust-remote-code

for name in \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    chat_template.jinja \
    generation_config.json \
    training_report.json \
    README.md; do
    if [[ -f "$HF_REFERENCE/$name" ]]; then
        cp "$HF_REFERENCE/$name" "$HF_PATH/$name"
    fi
done

echo "HF export: $HF_PATH"
