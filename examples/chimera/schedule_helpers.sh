#!/usr/bin/env bash

# Shared schedule arithmetic for Chimera launchers. Callers remain responsible
# for setting sensible stage defaults before invoking these helpers.

chimera_require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$name must be a positive integer, got: $value" >&2
        return 1
    }
}

chimera_require_fraction() {
    local name="$1"
    local value="$2"
    awk -v name="$name" -v value="$value" 'BEGIN {
        if (value !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ || value < 0 || value > 1) {
            printf "%s must be a number between 0 and 1, got: %s\n", name, value > "/dev/stderr"
            exit 1
        }
    }'
}

chimera_ceil_div() {
    local numerator="$1"
    local denominator="$2"
    echo $(( (numerator + denominator - 1) / denominator ))
}

chimera_ceil_fraction() {
    local total="$1"
    local fraction="$2"
    awk -v total="$total" -v fraction="$fraction" 'BEGIN {
        value = total * fraction
        rounded = int(value)
        if (value > rounded) rounded++
        if (rounded < 1) rounded = 1
        print rounded
    }'
}

chimera_scale_float() {
    local value="$1"
    local ratio="$2"
    awk -v value="$value" -v ratio="$ratio" 'BEGIN { printf "%.12g\n", value * ratio }'
}

chimera_resolve_schedule() {
    chimera_require_positive_integer TRAIN_ITERS "$TRAIN_ITERS"

    LR_DECAY_FRACTION="${LR_DECAY_FRACTION:-1.0}"
    LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.005}"
    LR_WSD_DECAY_FRACTION="${LR_WSD_DECAY_FRACTION:-0.10}"
    SAVE_INTERVAL_FRACTION="${SAVE_INTERVAL_FRACTION:-0.025}"
    EVAL_INTERVAL_FRACTION="${EVAL_INTERVAL_FRACTION:-0.005}"
    MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"

    chimera_require_fraction LR_DECAY_FRACTION "$LR_DECAY_FRACTION"
    chimera_require_fraction LR_WARMUP_FRACTION "$LR_WARMUP_FRACTION"
    chimera_require_fraction LR_WSD_DECAY_FRACTION "$LR_WSD_DECAY_FRACTION"
    chimera_require_fraction SAVE_INTERVAL_FRACTION "$SAVE_INTERVAL_FRACTION"
    chimera_require_fraction EVAL_INTERVAL_FRACTION "$EVAL_INTERVAL_FRACTION"
    chimera_require_fraction MIN_LR_RATIO "$MIN_LR_RATIO"

    LR_DECAY_ITERS="${LR_DECAY_ITERS:-$(chimera_ceil_fraction "$TRAIN_ITERS" "$LR_DECAY_FRACTION")}"
    LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-$(chimera_ceil_fraction "$TRAIN_ITERS" "$LR_WARMUP_FRACTION")}"
    LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$(chimera_ceil_fraction "$TRAIN_ITERS" "$LR_WSD_DECAY_FRACTION")}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-$(chimera_ceil_fraction "$TRAIN_ITERS" "$SAVE_INTERVAL_FRACTION")}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-$(chimera_ceil_fraction "$TRAIN_ITERS" "$EVAL_INTERVAL_FRACTION")}"
    MIN_LR="${MIN_LR:-$(chimera_scale_float "$LR" "$MIN_LR_RATIO")}"

    local name
    for name in LR_DECAY_ITERS LR_WARMUP_ITERS LR_WSD_DECAY_ITERS SAVE_INTERVAL EVAL_INTERVAL; do
        chimera_require_positive_integer "$name" "${!name}"
    done
    if (( LR_WARMUP_ITERS >= TRAIN_ITERS )); then
        LR_WARMUP_ITERS=$((TRAIN_ITERS > 1 ? TRAIN_ITERS - 1 : 0))
    fi
    if (( LR_DECAY_ITERS > TRAIN_ITERS )); then
        echo "LR_DECAY_ITERS=$LR_DECAY_ITERS cannot exceed TRAIN_ITERS=$TRAIN_ITERS" >&2
        return 1
    fi
    if (( LR_WSD_DECAY_ITERS > TRAIN_ITERS )); then
        echo "LR_WSD_DECAY_ITERS=$LR_WSD_DECAY_ITERS cannot exceed TRAIN_ITERS=$TRAIN_ITERS" >&2
        return 1
    fi
}
