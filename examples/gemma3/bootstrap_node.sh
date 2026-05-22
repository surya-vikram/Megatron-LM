#!/bin/bash
# bootstrap_node.sh: Automated startup script for new cloud instances (User Data)

echo "--- Initializing Gemma 3 Node Bootstrap ---"

# 1. Prepare persistent repository folder
mkdir -p /home/jovyan/repos && cd /home/jovyan/repos

# 2. Clone/Update the production branch if missing
if [ ! -d "Megatron-LM" ]; then
    echo "--- Cloning Production Repository ---"
    git clone -b gemma3 https://github.com/surya-vikram/Megatron-LM.git
fi

# 3. Enter the repo and run the modular setup
# This handles: 6-pillars, hopper kernels, bridge install, and load_env.sh
cd Megatron-LM
git checkout gemma3
bash examples/gemma3/setup.sh

# 4. Final verification and activation instruction
echo "=========================================================="
echo "GEMMA 3 BOOTSTRAP COMPLETE"
echo "To start working, run: source /home/jovyan/load_env.sh"
echo "=========================================================="
