#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HF_SOURCE=""
WORK_DIR=""
BRIDGE_PATH="/workspace/repos/Megatron-Bridge"
PYTHON_BIN=""

usage() {
    cat <<'USAGE'
Usage: bash examples/chimera/verify_conversion.sh --hf-source PATH --work-dir PATH [options]

Runs and verifies both conversion cycles:
  HF -> MCore -> HF, followed by MCore -> HF -> MCore -> HF.

Every HF endpoint must have exactly the expected key set, dtype, shape, tensor
bytes, and SHA256 digest. The work directory must not already exist.

Options:
  --hf-source PATH      Source HF Chimera checkpoint with safetensors weights.
  --work-dir PATH       New directory for intermediate checkpoints and reports.
  --bridge-path PATH    Megatron-Bridge repository path.
  --python PATH         Python executable.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-source) HF_SOURCE="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --bridge-path) BRIDGE_PATH="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

[[ -n "$HF_SOURCE" && -d "$HF_SOURCE" ]] || { echo "Missing --hf-source directory"; exit 1; }
[[ -n "$WORK_DIR" ]] || { echo "Missing --work-dir"; exit 1; }
[[ ! -e "$WORK_DIR" ]] || { echo "Work directory already exists: $WORK_DIR"; exit 1; }
[[ -d "$BRIDGE_PATH" ]] || { echo "Missing Megatron-Bridge repo: $BRIDGE_PATH"; exit 1; }

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /workspace/venv/bin/python ]]; then
        PYTHON_BIN="/workspace/venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

mkdir -p "$WORK_DIR"
MCORE_A="$WORK_DIR/hf_to_mcore"
HF_A="$WORK_DIR/hf_roundtrip"
MCORE_B="$WORK_DIR/mcore_roundtrip"
HF_B="$WORK_DIR/mcore_roundtrip_hf"

bash "$SCRIPT_DIR/import.sh" \
    --hf-model "$HF_SOURCE" \
    --mcore-path "$MCORE_A" \
    --bridge-path "$BRIDGE_PATH" \
    --python "$PYTHON_BIN"

bash "$SCRIPT_DIR/export.sh" \
    --hf-reference "$HF_SOURCE" \
    --mcore-path "$MCORE_A" \
    --hf-path "$HF_A" \
    --bridge-path "$BRIDGE_PATH" \
    --python "$PYTHON_BIN"

"$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" compare-hf \
    "$HF_SOURCE" "$HF_A" --report "$WORK_DIR/hf_to_mcore_to_hf_exact.json"

bash "$SCRIPT_DIR/import.sh" \
    --hf-model "$HF_A" \
    --mcore-path "$MCORE_B" \
    --bridge-path "$BRIDGE_PATH" \
    --python "$PYTHON_BIN"

bash "$SCRIPT_DIR/export.sh" \
    --hf-reference "$HF_A" \
    --mcore-path "$MCORE_B" \
    --hf-path "$HF_B" \
    --bridge-path "$BRIDGE_PATH" \
    --python "$PYTHON_BIN"

"$PYTHON_BIN" "$SCRIPT_DIR/architecture_contract.py" compare-hf \
    "$HF_A" "$HF_B" --report "$WORK_DIR/mcore_to_hf_to_mcore_to_hf_exact.json"

echo "Both exact Chimera conversion cycles passed: $WORK_DIR"
