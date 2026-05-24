#!/bin/bash
# cleanup_checkpoints.sh: Automated disk space guard for Megatron SFT training runs.
# Periodically monitors SFT checkpoints and purges older iterations to prevent disk OOM.

CHECKPOINT_DIR="/home/jovyan/data/checkpoints/gemma3-4b-sft"
INTERVAL_SECS=300  # check every 5 minutes

echo "[$(date)] Starting Megatron checkpoint disk guard..."
echo "[$(date)] Monitoring directory: $CHECKPOINT_DIR"

mkdir -p "$CHECKPOINT_DIR"

while true; do
    if [ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]; then
        LATEST=$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")
        # Strip any whitespace or newlines
        LATEST=$(echo "$LATEST" | tr -d '[:space:]')
        
        if [[ "$LATEST" =~ ^[0-9]+$ ]]; then
            for dir in "$CHECKPOINT_DIR"/iter_*; do
                if [ -d "$dir" ]; then
                    base_dir=$(basename "$dir")
                    iter_num=${base_dir#iter_}
                    # Strip leading zeros for decimal comparison
                    iter_num_clean=$(echo "$iter_num" | sed 's/^0*//')
                    [[ -z "$iter_num_clean" ]] && iter_num_clean=0
                    
                    if [ "$iter_num_clean" -lt "$LATEST" ]; then
                        echo "[$(date)] Disk Guard: Purging old checkpoint iteration $base_dir (Latest is $LATEST)"
                        rm -rf "$dir"
                    fi
                fi
            done
        fi
    fi
    sleep "$INTERVAL_SECS"
done
