#!/bin/bash
set -euo pipefail

# setup.sh: Environment Bootstrap for Gemma 3

echo "--- Initializing Gemma 3 Training Environment --- "

ROOT_DIR="/home/jovyan"
REPOS_DIR="$ROOT_DIR/repos"
MLM_DIR="$REPOS_DIR/Megatron-LM"
BRIDGE_DIR="$REPOS_DIR/Megatron-Bridge-Surya"

# 1. Base Container Setup (FlashAttn, TE, etc)
echo "--- Running Base Container Optimizations --- "
bash "$MLM_DIR/examples/gemma3/utils/setup_hopper_container.sh"

# 2. Megatron-Bridge Installation
echo "--- Installing Megatron-Bridge Dependency --- "
bash "$MLM_DIR/examples/gemma3/utils/install_bridge.sh"

# 3. Persistence Symlinks
echo "--- Configuring Persistent Symlinks --- "
rm -rf "$BRIDGE_DIR/3rdparty/Megatron-LM"
ln -s "$MLM_DIR" "$BRIDGE_DIR/3rdparty/Megatron-LM"

# 4. Generate load_env.sh for persistent activation
cat <<'EOE' > "$ROOT_DIR/load_env.sh"
export HF_HOME=/home/jovyan/models/.cache
export PYTHONPATH="/home/jovyan/repos/Megatron-LM:/home/jovyan/repos/Megatron-Bridge-Surya/src:$PYTHONPATH"
if [[ -d "/home/jovyan/venv" ]]; then
    source /home/jovyan/venv/bin/activate
fi
EOE

echo "--- Setup Complete --- "
echo "Please source $ROOT_DIR/load_env.sh to begin."
