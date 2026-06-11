#!/bin/bash
# compare_sft_simpo.sh — Run the same questions through SFT and SimPO models and print side-by-side output

REPO="/workspace/repos/Megatron-LM"
SFT_MODEL="/datasets/megadata/hf_models/gemma-3-1b-sft-hf"
SIMPO_MODEL="/datasets/megadata/hf_models/gemma-3-1b-simpo-hf"
INFER="python3 ${REPO}/examples/gemma3/infer.py"

cd "$REPO"

QUESTIONS=(
  "Explain the water cycle in 2 sentences."
  "What is 17 multiplied by 13?"
  "Write a haiku about mountains."
)

run_model() {
  local label="$1"
  local model="$2"
  echo ""
  echo "========================================"
  echo "  $label"
  echo "========================================"
  for Q in "${QUESTIONS[@]}"; do
    echo ""
    echo ">>> Q: $Q"
    echo "$Q" | $INFER --model "$model" --chat --max-new-tokens 150 2>/dev/null
    echo ""
  done
}

run_model "SFT MODEL" "$SFT_MODEL"
run_model "SIMPO MODEL" "$SIMPO_MODEL"

echo ""
echo "=== DONE ==="
