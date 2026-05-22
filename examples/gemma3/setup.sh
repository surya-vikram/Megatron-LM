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
mkdir -p "$ROOT_DIR"/{models,data,logs,offline_tools,repos}

# 2. Create persistent venv if not exists
if [[ ! -f "$ROOT_DIR/venv/bin/activate" ]]; then
    echo "--- Creating Virtual Environment --- "
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH=$HOME/.local/bin:/root/.local/bin:/home/jovyan/.local/bin:$PATH
    fi
    mkdir -p "$ROOT_DIR/venv"
    uv venv "$ROOT_DIR/venv" --system-site-packages
fi

# 3. Base Container Setup
echo "--- Running Base Container Optimizations --- "
bash "$MLM_DIR/examples/gemma3/utils/setup_hopper_container.sh"

# 4. Megatron-Bridge Installation
echo "--- Installing Megatron-Bridge --- "
bash "$MLM_DIR/examples/gemma3/utils/install_bridge.sh"

# 5. Production Extra Dependencies
echo "--- Installing Data Ingestion & Monitoring Tools --- "
source "$ROOT_DIR/venv/bin/activate"
pip install datasets tqdm wandb --quiet

# 6. Persistence Symlinks
echo "--- Configuring Persistent Symlinks --- "
mkdir -p "$BRIDGE_DIR/3rdparty"
rm -rf "$BRIDGE_DIR/3rdparty/Megatron-LM"
ln -s "$MLM_DIR" "$BRIDGE_DIR/3rdparty/Megatron-LM"

# 7. Generate load_env.sh
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
