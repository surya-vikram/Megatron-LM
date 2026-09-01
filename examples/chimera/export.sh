#!/bin/bash
set -euo pipefail

HF_REFERENCE="/datasets/megadata/hf_models/chimera-10b"
MCORE_PATH="/datasets/megadata/chimera_runs/overfit/checkpoints"
HF_PATH="/datasets/megadata/hf_exports/chimera-overfit-hf"
BRIDGE_PATH="/workspace/repos/Megatron-Bridge"
PYTHON_BIN=""
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOAD_WITH_BIAS=""

usage() {
    cat <<'USAGE'
Usage: bash examples/chimera/export.sh [options]

Options:
  --hf-reference PATH   Original HF Chimera model directory used as config/tokenizer reference.
  --mcore-path PATH     Trained Megatron-Core checkpoint root or iteration directory.
  --hf-path PATH        Output HF model directory.
  --bridge-path PATH    Megatron-Bridge repository path.
  --load-with-bias      Export config applies the frozen router correction bias.
  --no-load-with-bias   Export config bypasses the frozen bias but preserves its tensor.
  --python PATH         Python executable.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-reference) HF_REFERENCE="$2"; shift 2 ;;
        --mcore-path) MCORE_PATH="$2"; shift 2 ;;
        --hf-path) HF_PATH="$2"; shift 2 ;;
        --bridge-path) BRIDGE_PATH="$2"; shift 2 ;;
        --load-with-bias) LOAD_WITH_BIAS="true"; shift ;;
        --no-load-with-bias) LOAD_WITH_BIAS="false"; shift ;;
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

"$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" validate-hf "$HF_REFERENCE"

CHECKPOINT_ROOT="$MCORE_PATH"
CHECKPOINT_ITER=""
if [[ "$(basename "$MCORE_PATH")" == iter_* ]]; then
    CHECKPOINT_ROOT=$(dirname "$MCORE_PATH")
    CHECKPOINT_ITER="$MCORE_PATH"
else
    CHECKPOINT_ITER=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'iter_*' | sort -V | tail -n 1)
fi

[[ -n "$CHECKPOINT_ITER" ]] || {
    echo "No iter_* checkpoint directory found under $CHECKPOINT_ROOT." >&2
    exit 1
}

ITER_RUN_CONFIG="$CHECKPOINT_ITER/run_config.yaml"
ROOT_RUN_CONFIG="$CHECKPOINT_ROOT/run_config.yaml"
[[ -f "$ITER_RUN_CONFIG" || -f "$ROOT_RUN_CONFIG" ]] || {
    echo "Missing checkpoint-generated run_config.yaml for selected iteration: $CHECKPOINT_ITER" >&2
    exit 1
}

if [[ -f "$ITER_RUN_CONFIG" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" validate-run-config "$ITER_RUN_CONFIG"
    RUN_CONFIG="$ITER_RUN_CONFIG"
else
    "$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" validate-run-config "$ROOT_RUN_CONFIG"
    RUN_CONFIG="$ROOT_RUN_CONFIG"
fi
if [[ -f "$ITER_RUN_CONFIG" && -f "$ROOT_RUN_CONFIG" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" validate-run-config "$ROOT_RUN_CONFIG"
    cmp -s "$ROOT_RUN_CONFIG" "$ITER_RUN_CONFIG" || {
        echo "Checkpoint root and iteration run_config.yaml files differ." >&2
        exit 1
    }
fi
echo "Using checkpoint-generated run config: $RUN_CONFIG"

mkdir -p "$(dirname "$HF_PATH")"

export PYTHONPATH="$BRIDGE_PATH/src:${PYTHONPATH:-}"

"$PYTHON_BIN" "$BRIDGE_PATH/examples/conversion/convert_checkpoints.py" export \
    --hf-model "$HF_REFERENCE" \
    --megatron-path "$CHECKPOINT_ITER" \
    --hf-path "$HF_PATH" \
    --trust-remote-code

if [[ -n "$LOAD_WITH_BIAS" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" set-load-with-bias "$HF_PATH" "$LOAD_WITH_BIAS"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" validate-hf "$HF_PATH" --weights

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
