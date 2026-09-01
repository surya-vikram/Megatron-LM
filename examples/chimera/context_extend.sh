#!/bin/bash
set -euo pipefail

# Continue a completed Chimera phase into the next canonical YaRN context.
# This intentionally loads model weights only and starts a fresh Adam optimizer
# and RNG stream for the new data/schedule phase.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONTEXT_PHASE="${CONTEXT_PHASE:-}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"

case "$CONTEXT_PHASE" in
    32k)
        PREVIOUS_CONTEXT_PHASE=8k
        DEFAULT_CP_SIZE=4
        DEFAULT_GLOBAL_BATCH_SIZE=144
        ;;
    64k)
        PREVIOUS_CONTEXT_PHASE=32k
        DEFAULT_CP_SIZE=8
        DEFAULT_GLOBAL_BATCH_SIZE=72
        ;;
    128k)
        PREVIOUS_CONTEXT_PHASE=64k
        DEFAULT_CP_SIZE=16
        DEFAULT_GLOBAL_BATCH_SIZE=36
        ;;
    *)
        echo "Set CONTEXT_PHASE to the next extension phase: 32k, 64k, or 128k" >&2
        exit 1
        ;;
esac

source "$SCRIPT_DIR/context_phase.sh"
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
    --profile full --context-phase "$PREVIOUS_CONTEXT_PHASE"

SEQ_LENGTH="$MAX_POSITION_EMBEDDINGS"
CP_SIZE="${CP_SIZE:-$DEFAULT_CP_SIZE}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$DEFAULT_GLOBAL_BATCH_SIZE}"
EXTENSION_TOKENS="${EXTENSION_TOKENS:-10000000000}"
TOKENS_PER_ITER=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS="${TRAIN_ITERS:-$(( (EXTENSION_TOKENS + TOKENS_PER_ITER - 1) / TOKENS_PER_ITER ))}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-$(( (TRAIN_ITERS + 9) / 10 ))}"

CHIMERA_CONTEXT_EXTENSION=true
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
RUNS_ROOT="${RUNS_ROOT:-/datasets/megadata/chimera_context_${CONTEXT_PHASE}_runs}"

export CONTEXT_PHASE LOAD_CHECKPOINT SEQ_LENGTH CP_SIZE MICRO_BATCH_SIZE
export GLOBAL_BATCH_SIZE TRAIN_ITERS LR_WARMUP_ITERS CHIMERA_CONTEXT_EXTENSION
export LR MIN_LR LR_DECAY_STYLE LR_DECAY_ITERS WEIGHT_DECAY
export SAVE_INTERVAL EVAL_INTERVAL RUNS_ROOT EXTENSION_TOKENS

echo "Launching Chimera context extension"
echo "  Transition:       $PREVIOUS_CONTEXT_PHASE -> $CONTEXT_PHASE"
echo "  Source checkpoint: $LOAD_CHECKPOINT"
echo "  YaRN geometry:    max=$MAX_POSITION_EMBEDDINGS factor=$ROTARY_SCALING_FACTOR original=$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS"
echo "  Seq/CP/batch:     seq=$SEQ_LENGTH CP=$CP_SIZE micro=$MICRO_BATCH_SIZE global=$GLOBAL_BATCH_SIZE"
echo "  Budget/schedule:  target_tokens=$EXTENSION_TOKENS tokens_per_iter=$TOKENS_PER_ITER iters=$TRAIN_ITERS warmup=$LR_WARMUP_ITERS"

exec "$SCRIPT_DIR/train.sh"
