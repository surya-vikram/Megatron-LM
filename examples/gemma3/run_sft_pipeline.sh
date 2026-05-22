#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./examples/gemma3/run_sft_pipeline.sh --stage <prepare|preflight|overfit|smoke|launch|export|evaluate|full> [options]

Key options:
  --hf-token <token>                HuggingFace token. Falls back to $HF_TOKEN.
  --hf-model <model>                Base pretrained HF model path/ID. Default: google/gemma-3-4b-pt
  --hf-config-ref <model>           HF config/tokenizer ref for export/eval. Default: google/gemma-3-4b-it
  --output-root <dir>               Root output directory. Default: /home/jovyan/models/gemma3_sft_runs
  --run-name <name>                 Run directory name. Default: timestamp
  --data-bundle-dir <dir>           Bundle dir with manifest.json or prepared jsonl files.
  --train-data-path <path>          Real corpus train jsonl. Optional if bundle provides train.jsonl.
  --heldout-path <path>             Held-out eval jsonl. Optional if bundle provides heldout.jsonl.
  --overfit-single-path <path>      One-sample overfit jsonl. Optional if bundle provides it.
  --overfit-pack-path <path>        Tiny-pack overfit jsonl. Optional if bundle provides it.
  --reasoning-eval-path <path>      Reasoning eval json. Optional if bundle provides it.
  --base-checkpoint <path>          Megatron base checkpoint directory.
  --context-ladder "a b c"          Default: 16384 24576 32768
  --backoff-ladder "a b"            Default: 12288 8192
  --master-port-base <port>         Base rendezvous port. Default: 6000
EOF
}

STAGE="full"
HF_TOKEN_ARG="${HF_TOKEN:-}"
HF_MODEL="google/gemma-3-4b-pt"
HF_CONFIG_REF="google/gemma-3-4b-it"
OUTPUT_ROOT="/home/jovyan/models/gemma3_sft_runs"
RUN_NAME="$(date +%Y%m%d_%H%M%S)"
DATA_BUNDLE_DIR=""
TRAIN_DATA_PATH=""
HELDOUT_PATH=""
OVERFIT_SINGLE_PATH=""
OVERFIT_PACK_PATH=""
REASONING_EVAL_PATH=""
BASE_CHECKPOINT=""
CONTEXT_LADDER="16384 24576 32768"
BACKOFF_LADDER="12288 8192"
PROBE_ITERS=10
OVERFIT_TRAIN_ITERS=100
SMOKE_TRAIN_ITERS=150
LAUNCH_TRAIN_ITERS=100
OVERFIT_LR="1e-5"
SMOKE_LR="5e-6"
FULL_LR="3e-6"
GLOBAL_BATCH_SIZE=8
NUM_WORKERS=8
MASTER_PORT_BASE=6000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN_ARG="$2"
            shift 2
            ;;
        --hf-model)
            HF_MODEL="$2"
            shift 2
            ;;
        --hf-config-ref)
            HF_CONFIG_REF="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --data-bundle-dir)
            DATA_BUNDLE_DIR="$2"
            shift 2
            ;;
        --train-data-path)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --heldout-path)
            HELDOUT_PATH="$2"
            shift 2
            ;;
        --overfit-single-path)
            OVERFIT_SINGLE_PATH="$2"
            shift 2
            ;;
        --overfit-pack-path)
            OVERFIT_PACK_PATH="$2"
            shift 2
            ;;
        --reasoning-eval-path)
            REASONING_EVAL_PATH="$2"
            shift 2
            ;;
        --base-checkpoint)
            BASE_CHECKPOINT="$2"
            shift 2
            ;;
        --context-ladder)
            CONTEXT_LADDER="$2"
            shift 2
            ;;
        --backoff-ladder)
            BACKOFF_LADDER="$2"
            shift 2
            ;;
        --probe-iters)
            PROBE_ITERS="$2"
            shift 2
            ;;
        --overfit-train-iters)
            OVERFIT_TRAIN_ITERS="$2"
            shift 2
            ;;
        --smoke-train-iters)
            SMOKE_TRAIN_ITERS="$2"
            shift 2
            ;;
        --launch-train-iters)
            LAUNCH_TRAIN_ITERS="$2"
            shift 2
            ;;
        --overfit-lr)
            OVERFIT_LR="$2"
            shift 2
            ;;
        --smoke-lr)
            SMOKE_LR="$2"
            shift 2
            ;;
        --full-lr)
            FULL_LR="$2"
            shift 2
            ;;
        --global-batch-size)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --master-port-base)
            MASTER_PORT_BASE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$BASE_CHECKPOINT" ]]; then
    # Sanitize HF_MODEL name for directory path
    M_NAME=$(basename "$HF_MODEL")
    BASE_CHECKPOINT="/home/jovyan/models/${M_NAME}-mcore-sft-base"
fi

export HF_TOKEN="$HF_TOKEN_ARG"
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:/home/jovyan/Megatron-Bridge/src"

WORK_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT_ROOT="${WORK_DIR}/checkpoints"
PROBE_ROOT="${WORK_DIR}/probes"
CONTEXT_FILE="${WORK_DIR}/chosen_context.txt"
DATA_BUNDLE_DIR="${DATA_BUNDLE_DIR:-${WORK_DIR}/data_bundle}"
OVERFIT_SAVE_PATH="${CHECKPOINT_ROOT}/overfit_single"
SMOKE_SAVE_PATH="${CHECKPOINT_ROOT}/smoke"
FULL_SAVE_PATH="${CHECKPOINT_ROOT}/full"
EXPORT_PATH="${WORK_DIR}/exported_hf"
OVERFIT_EXPORT_PATH="${WORK_DIR}/exported_overfit"
SMOKE_EXPORT_PATH="${WORK_DIR}/exported_smoke"

mkdir -p "$WORK_DIR" "$CHECKPOINT_ROOT" "$PROBE_ROOT"

require_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "Warning: HF_TOKEN is not set. Proceeding with public model access only."
    fi
}

log_boundary() {
    echo "===================================================="
    echo "$1"
    echo "===================================================="
}

maybe_import_base_checkpoint() {
    if [[ -f "${BASE_CHECKPOINT}/latest_checkpointed_iteration.txt" ]]; then
        echo "Base Megatron checkpoint already exists: ${BASE_CHECKPOINT}"
        return 0
    fi
    require_hf_token
    log_boundary "Importing Base Checkpoint"
    ./examples/gemma3/01_convert_mcore.sh "$HF_MODEL" "$BASE_CHECKPOINT"
}

prepare_smoke_bundle() {
    log_boundary "Preparing SFT Bundle"
    python3 examples/gemma3/prepare_sft_data.py \
        --output-dir "$DATA_BUNDLE_DIR" \
        --tokenizer-model "$HF_CONFIG_REF" \
        --max-seq-length 32768 \
        --shuffle
}

bundle_value() {
    local key="$1"
    python3 - <<PY
import json
from pathlib import Path

manifest = Path("${DATA_BUNDLE_DIR}") / "manifest.json"
if manifest.exists():
    payload = json.loads(manifest.read_text())
    print(payload["paths"].get("${key}", ""))
PY
}

resolve_eval_paths() {
    if [[ -z "$TRAIN_DATA_PATH" ]]; then
        TRAIN_DATA_PATH="$(bundle_value train)"
    fi
    if [[ -z "$HELDOUT_PATH" ]]; then
        HELDOUT_PATH="$(bundle_value heldout)"
    fi
    if [[ -z "$OVERFIT_SINGLE_PATH" ]]; then
        OVERFIT_SINGLE_PATH="$(bundle_value overfit_single)"
    fi
    if [[ -z "$OVERFIT_PACK_PATH" ]]; then
        OVERFIT_PACK_PATH="$(bundle_value overfit_pack)"
    fi
    if [[ -z "$REASONING_EVAL_PATH" ]]; then
        REASONING_EVAL_PATH="$(bundle_value reasoning_eval)"
    fi
}

resolve_smoke_train_path() {
    local smoke_path
    smoke_path="$(bundle_value smoke_train)"
    if [[ -n "$smoke_path" ]]; then
        echo "$smoke_path"
        return 0
    fi
    echo "$TRAIN_DATA_PATH"
}

probe_context() {
    local context="$1"
    local master_port="$2"
    local probe_save="${PROBE_ROOT}/seq_${context}"
    local log_path="${PROBE_ROOT}/seq_${context}.log"
    mkdir -p "$probe_save"
    echo "Probing context length ${context} ..."
    if ./examples/gemma3/04_sft_mcore.sh \
        --checkpoint-path "$BASE_CHECKPOINT" \
        --data-path "$TRAIN_DATA_PATH" \
        --save-path "$probe_save" \
        --mode "preflight_seq_${context}" \
        --seq-length "$context" \
        --master-port "$master_port" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --train-iters "$PROBE_ITERS" \
        --lr "$SMOKE_LR" \
        --num-workers "$NUM_WORKERS" \
        --save-interval "$PROBE_ITERS" \
        --eval-iters 0 \
        --eval-interval 100000 >"$log_path" 2>&1; then
        echo "Context ${context} passed."
        return 0
    fi
    echo "Context ${context} failed. Inspect ${log_path}"
    return 1
}

run_context_preflight() {
    resolve_eval_paths
    if [[ -z "$TRAIN_DATA_PATH" ]]; then
        echo "Error: train data path is required for context preflight."
        exit 1
    fi

    local chosen=""
    local primary_success=1
    for context in $CONTEXT_LADDER; do
        if probe_context "$context" "$MASTER_PORT_BASE"; then
            chosen="$context"
        else
            primary_success=0
            break
        fi
    done

    if [[ "$primary_success" -eq 0 && -z "$chosen" ]]; then
        for context in $BACKOFF_LADDER; do
            if probe_context "$context" "$MASTER_PORT_BASE"; then
                chosen="$context"
                break
            fi
        done
    fi

    if [[ -z "$chosen" ]]; then
        echo "Error: no stable context length found."
        exit 1
    fi

    echo "$chosen" >"$CONTEXT_FILE"
    log_boundary "Chosen Stable Context Length: ${chosen}"
}

require_context_choice() {
    if [[ ! -f "$CONTEXT_FILE" ]]; then
        run_context_preflight
    fi
    cat "$CONTEXT_FILE"
}

run_train_stage() {
    local mode="$1"
    local data_path="$2"
    local save_path="$3"
    local load_path="$4"
    local seq_length="$5"
    local train_iters="$6"
    local lr="$7"
    local master_port="$8"

    ./examples/gemma3/04_sft_mcore.sh \
        --checkpoint-path "$load_path" \
        --data-path "$data_path" \
        --save-path "$save_path" \
        --mode "$mode" \
        --seq-length "$seq_length" \
        --master-port "$master_port" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --train-iters "$train_iters" \
        --lr "$lr" \
        --num-workers "$NUM_WORKERS"
}

run_export_stage() {
    local checkpoint_path="$1"
    local export_path="$2"
    require_hf_token
    ./examples/gemma3/03_export_mcore.sh "$checkpoint_path" "$export_path" "$HF_CONFIG_REF"
}

run_verify_stage() {
    local verification_mode="$1"
    local model_path="$2"
    local run_config="$3"
    python3 examples/gemma3/verify_sft_results.py \
        --verification-mode "$verification_mode" \
        --model-path "$model_path" \
        --base-model-id "$HF_CONFIG_REF" \
        --data-bundle-dir "$DATA_BUNDLE_DIR" \
        --train-data-path "$TRAIN_DATA_PATH" \
        --heldout-path "$HELDOUT_PATH" \
        --overfit-single-path "$OVERFIT_SINGLE_PATH" \
        --overfit-pack-path "$OVERFIT_PACK_PATH" \
        --reasoning-eval-path "$REASONING_EVAL_PATH" \
        --run-config "$run_config"
}

run_overfit_gate() {
    resolve_eval_paths
    local seq_length
    seq_length="$(require_context_choice)"
    if [[ -z "$OVERFIT_SINGLE_PATH" ]]; then
        echo "Error: overfit single data path is required."
        exit 1
    fi
    log_boundary "Running 1-Sample Overfit Gate"
    run_train_stage "overfit_single" "$OVERFIT_SINGLE_PATH" "$OVERFIT_SAVE_PATH" "$BASE_CHECKPOINT" "$seq_length" "$OVERFIT_TRAIN_ITERS" "$OVERFIT_LR" "$((MASTER_PORT_BASE + 1))"
    run_export_stage "$OVERFIT_SAVE_PATH" "$OVERFIT_EXPORT_PATH"
    run_verify_stage "overfit" "$OVERFIT_EXPORT_PATH" "${OVERFIT_SAVE_PATH}/run_config.json"
}

run_smoke_gate() {
    resolve_eval_paths
    local seq_length
    seq_length="$(require_context_choice)"
    local smoke_data
    smoke_data="$(resolve_smoke_train_path)"
    if [[ -z "$smoke_data" ]]; then
        echo "Error: smoke data path is required."
        exit 1
    fi
    log_boundary "Running Smoke Gate"
    run_train_stage "smoke" "$smoke_data" "$SMOKE_SAVE_PATH" "$BASE_CHECKPOINT" "$seq_length" "$SMOKE_TRAIN_ITERS" "$SMOKE_LR" "$((MASTER_PORT_BASE + 2))"
    run_export_stage "$SMOKE_SAVE_PATH" "$SMOKE_EXPORT_PATH"
    run_verify_stage "smoke" "$SMOKE_EXPORT_PATH" "${SMOKE_SAVE_PATH}/run_config.json"
}

run_launch_stage() {
    resolve_eval_paths
    local seq_length
    seq_length="$(require_context_choice)"
    if [[ -z "$TRAIN_DATA_PATH" ]]; then
        echo "Error: full launch requires train data."
        exit 1
    fi
    log_boundary "Launching Real-Corpus SFT"
    run_train_stage "full" "$TRAIN_DATA_PATH" "$FULL_SAVE_PATH" "$BASE_CHECKPOINT" "$seq_length" "$LAUNCH_TRAIN_ITERS" "$FULL_LR" "$((MASTER_PORT_BASE + 3))"
}

run_full_export_and_eval() {
    resolve_eval_paths
    log_boundary "Exporting Final Checkpoint"
    run_export_stage "$FULL_SAVE_PATH" "$EXPORT_PATH"
    log_boundary "Running Stronger Evaluation"
    run_verify_stage "full" "$EXPORT_PATH" "${FULL_SAVE_PATH}/run_config.json"
}

case "$STAGE" in
    prepare)
        prepare_smoke_bundle
        ;;
    preflight)
        maybe_import_base_checkpoint
        if [[ ! -f "${DATA_BUNDLE_DIR}/manifest.json" && -z "$TRAIN_DATA_PATH" ]]; then
            prepare_smoke_bundle
        fi
        run_context_preflight
        ;;
    overfit)
        maybe_import_base_checkpoint
        if [[ ! -f "${DATA_BUNDLE_DIR}/manifest.json" && -z "$OVERFIT_SINGLE_PATH" ]]; then
            prepare_smoke_bundle
        fi
        run_overfit_gate
        ;;
    smoke)
        maybe_import_base_checkpoint
        if [[ ! -f "${DATA_BUNDLE_DIR}/manifest.json" && -z "$TRAIN_DATA_PATH" ]]; then
            prepare_smoke_bundle
        fi
        run_smoke_gate
        ;;
    launch)
        maybe_import_base_checkpoint
        run_launch_stage
        ;;
    export)
        run_full_export_and_eval
        ;;
    evaluate)
        resolve_eval_paths
        run_verify_stage "full" "$EXPORT_PATH" "${FULL_SAVE_PATH}/run_config.json"
        ;;
    full)
        maybe_import_base_checkpoint
        if [[ ! -f "${DATA_BUNDLE_DIR}/manifest.json" && -z "$TRAIN_DATA_PATH" ]]; then
            prepare_smoke_bundle
        fi
        run_context_preflight
        run_overfit_gate
        run_smoke_gate
        run_launch_stage
        run_full_export_and_eval
        ;;
    *)
        echo "Unknown stage: ${STAGE}"
        usage
        exit 1
        ;;
esac
