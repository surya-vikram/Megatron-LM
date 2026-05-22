#!/bin/bash
set -euo pipefail

# setup.sh: Comprehensive Environment Bootstrap for Gemma 3

echo "--- Initializing Gemma 3 Training Environment --- "

ROOT_DIR="/home/jovyan"
REPOS_DIR="$ROOT_DIR/repos"
MLM_DIR="$REPOS_DIR/Megatron-LM"
BRIDGE_DIR="$REPOS_DIR/Megatron-Bridge"

# 1. Create the 6 Pillars of Persistence
echo "--- Ensuring Persistent Directories Exist --- "
mkdir -p "$ROOT_DIR"/{models,data,logs,offline_tools,venv,repos}

# 2. Base Container Setup
echo "--- Running Base Container Optimizations --- "
bash "$MLM_DIR/examples/gemma3/utils/setup_hopper_container.sh"

# 3. Megatron-Bridge Installation (Handles cloning internally)
echo "--- Installing Megatron-Bridge --- "
bash "$MLM_DIR/examples/gemma3/utils/install_bridge.sh"

# 4. Production Extra Dependencies
echo "--- Installing Data Ingestion & Monitoring Tools --- "
pip install datasets tqdm wandb --quiet

# 5. Persistence Symlinks
echo "--- Configuring Persistent Symlinks --- "
mkdir -p "$BRIDGE_DIR/3rdparty"
rm -rf "$BRIDGE_DIR/3rdparty/Megatron-LM"
ln -s "$MLM_DIR" "$BRIDGE_DIR/3rdparty/Megatron-LM"

# 6. Generate load_env.sh
echo "--- Generating Environment Activation Key --- "
cat <<'EOE' > "$ROOT_DIR/load_env.sh"
export HF_HOME=/home/jovyan/models/.cache
export PYTHONPATH="/home/jovyan/repos/Megatron-LM:/home/jovyan/repos/Megatron-Bridge/src:$PYTHONPATH"
if [[ -d "/home/jovyan/venv" ]]; then
    source /home/jovyan/venv/bin/activate
fi
EOE

echo "--- Setup Complete --- "
echo "Please run: source $ROOT_DIR/load_env.sh"
