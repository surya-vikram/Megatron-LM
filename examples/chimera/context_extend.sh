#!/bin/bash
set -euo pipefail

# Continue a completed Chimera phase into any longer canonical YaRN context.
# This intentionally loads model weights only and starts a fresh optimizer
# and RNG stream for the new data/schedule phase.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONTEXT_PHASE="${CONTEXT_PHASE:-}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"

case "$CONTEXT_PHASE" in
    32k|64k|128k)
        DEFAULT_CP_SIZE=1
        DEFAULT_GLOBAL_BATCH_SIZE=48
        ;;
    *)
        echo "Set CONTEXT_PHASE to the next extension phase: 32k, 64k, or 128k" >&2
        exit 1
        ;;
esac

source "$SCRIPT_DIR/context_phase.sh"
source "$SCRIPT_DIR/schedule_helpers.sh"
[[ -n "$LOAD_CHECKPOINT" ]] || { echo "Context extension requires LOAD_CHECKPOINT" >&2; exit 1; }
[[ -d "$LOAD_CHECKPOINT" ]] || { echo "Missing checkpoint root: $LOAD_CHECKPOINT" >&2; exit 1; }

SOURCE_RUN_CONFIG="$LOAD_CHECKPOINT/run_config.yaml"
if [[ ! -f "$SOURCE_RUN_CONFIG" ]]; then
    SOURCE_RUN_CONFIG=$(find "$LOAD_CHECKPOINT" -name run_config.yaml -type f -print -quit)
fi
[[ -n "$SOURCE_RUN_CONFIG" && -f "$SOURCE_RUN_CONFIG" ]] || {
    echo "Missing checkpoint run_config.yaml under $LOAD_CHECKPOINT" >&2; exit 1;
}
python3 "$SCRIPT_DIR/architecture_contract.py" validate-run-config "$SOURCE_RUN_CONFIG" \
    --profile full --context-phase auto
PREVIOUS_CONTEXT_PHASE=$(python3 - "$SOURCE_RUN_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(config["model"]["chimera_context_phase"])
PY
)
case "$PREVIOUS_CONTEXT_PHASE:$CONTEXT_PHASE" in
    8k:32k|8k:64k|8k:128k|32k:64k|32k:128k|64k:128k) ;;
    *)
        echo "Invalid context transition: $PREVIOUS_CONTEXT_PHASE -> $CONTEXT_PHASE" >&2
        exit 1
        ;;
esac

SEQ_LENGTH="$MAX_POSITION_EMBEDDINGS"
CP_SIZE="${CP_SIZE:-$DEFAULT_CP_SIZE}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$DEFAULT_GLOBAL_BATCH_SIZE}"
TRAIN_TOKENS="${TRAIN_TOKENS:-${EXTENSION_TOKENS:-10000000000}}"
TOKENS_PER_ITER=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS="${TRAIN_ITERS:-$(chimera_ceil_div "$TRAIN_TOKENS" "$TOKENS_PER_ITER")}"

CHIMERA_CONTEXT_EXTENSION=true
LR="${LR:-1e-5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.10}"
LR_DECAY_FRACTION="${LR_DECAY_FRACTION:-1.0}"
LR_WSD_DECAY_FRACTION="${LR_WSD_DECAY_FRACTION:-0.10}"
SAVE_INTERVAL_FRACTION="${SAVE_INTERVAL_FRACTION:-0.25}"
EVAL_INTERVAL_FRACTION="${EVAL_INTERVAL_FRACTION:-0.10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_context_${CONTEXT_PHASE}_runs}"

chimera_resolve_schedule

export CONTEXT_PHASE LOAD_CHECKPOINT SEQ_LENGTH CP_SIZE MICRO_BATCH_SIZE
export GLOBAL_BATCH_SIZE TRAIN_ITERS LR_WARMUP_ITERS CHIMERA_CONTEXT_EXTENSION
export LR MIN_LR LR_DECAY_STYLE LR_DECAY_ITERS WEIGHT_DECAY
export SAVE_INTERVAL EVAL_INTERVAL RUNS_ROOT TRAIN_TOKENS MIN_LR MIN_LR_RATIO
export LR_WARMUP_FRACTION LR_DECAY_FRACTION LR_WSD_DECAY_FRACTION
export SAVE_INTERVAL_FRACTION EVAL_INTERVAL_FRACTION LR_WSD_DECAY_ITERS

echo "Launching Chimera context extension"
echo "  Transition:       $PREVIOUS_CONTEXT_PHASE -> $CONTEXT_PHASE"
echo "  Source checkpoint: $LOAD_CHECKPOINT"
echo "  YaRN geometry:    max=$MAX_POSITION_EMBEDDINGS factor=$ROTARY_SCALING_FACTOR original=$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"
echo "  Seq/CP/batch:     seq=$SEQ_LENGTH CP=$CP_SIZE micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE"
echo "  Budget/schedule:  target_tokens=$TRAIN_TOKENS tokens_per_iter=$TOKENS_PER_ITER iters=$TRAIN_ITERS warmup=$LR_WARMUP_ITERS"

exec "$SCRIPT_DIR/train.sh"
