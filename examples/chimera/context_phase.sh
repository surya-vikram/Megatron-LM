#!/bin/bash

# Resolve the immutable YaRN geometry for one Chimera context phase.
# Callers may choose a shorter SEQ_LENGTH for smoke tests or short SFT data,
# but may not override the phase's maximum, factor, or original context.

CONTEXT_PHASE="${CONTEXT_PHASE:-8k}"

case "$CONTEXT_PHASE" in
    8k)
        CHIMERA_CONTEXT_MAX=8192
        CHIMERA_YARN_FACTOR=1.0
        ;;
    32k)
        CHIMERA_CONTEXT_MAX=32768
        CHIMERA_YARN_FACTOR=4.0
        ;;
    64k)
        CHIMERA_CONTEXT_MAX=65536
        CHIMERA_YARN_FACTOR=8.0
        ;;
    128k)
        CHIMERA_CONTEXT_MAX=131072
        CHIMERA_YARN_FACTOR=16.0
        ;;
    *)
        echo "Unsupported CONTEXT_PHASE=$CONTEXT_PHASE; expected 8k, 32k, 64k, or 128k" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

if [[ -n "${POSITION_EMBEDDING_TYPE:-}" && "$POSITION_EMBEDDING_TYPE" != yarn ]]; then
    echo "Chimera supports only POSITION_EMBEDDING_TYPE=yarn" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ -n "${MAX_POSITION_EMBEDDINGS:-}" && "$MAX_POSITION_EMBEDDINGS" -ne "$CHIMERA_CONTEXT_MAX" ]]; then
    echo "CONTEXT_PHASE=$CONTEXT_PHASE requires MAX_POSITION_EMBEDDINGS=$CHIMERA_CONTEXT_MAX" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ -n "${ROTARY_SCALING_FACTOR:-}" \
    && "$ROTARY_SCALING_FACTOR" != "$CHIMERA_YARN_FACTOR" \
    && "$ROTARY_SCALING_FACTOR" != "${CHIMERA_YARN_FACTOR%.0}" ]]; then
    echo "CONTEXT_PHASE=$CONTEXT_PHASE requires ROTARY_SCALING_FACTOR=$CHIMERA_YARN_FACTOR" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ -n "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-}" && "$YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS" -ne 8192 ]]; then
    echo "Chimera requires YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=8192" >&2
    return 1 2>/dev/null || exit 1
fi

POSITION_EMBEDDING_TYPE=yarn
MAX_POSITION_EMBEDDINGS=$CHIMERA_CONTEXT_MAX
ROTARY_SCALING_FACTOR=$CHIMERA_YARN_FACTOR
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=8192
SEQ_LENGTH="${SEQ_LENGTH:-$MAX_POSITION_EMBEDDINGS}"

if (( SEQ_LENGTH <= 0 || SEQ_LENGTH > MAX_POSITION_EMBEDDINGS )); then
    echo "SEQ_LENGTH=$SEQ_LENGTH must be between 1 and $MAX_POSITION_EMBEDDINGS for phase $CONTEXT_PHASE" >&2
    return 1 2>/dev/null || exit 1
fi

export CONTEXT_PHASE POSITION_EMBEDDING_TYPE MAX_POSITION_EMBEDDINGS
export ROTARY_SCALING_FACTOR YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS SEQ_LENGTH
