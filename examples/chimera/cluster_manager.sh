#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/schedule_helpers.sh"

# Variables passed unchanged from the selected stage env file into every rank.
FORWARDED_ENV_VARS=(
    RUNS_ROOT TOKENIZER_MODEL TRAIN_DATA_PATH VALID_DATA_PATH LOAD_CHECKPOINT
    DATA_PATH MCORE_PATH INTRA_DOC_MASKING CONTEXT_PHASE SEQ_LENGTH
    TP_SIZE PP_SIZE EP_SIZE CP_SIZE MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE
    TRAIN_TOKENS TRAIN_ITERS TRAIN_EPOCHS LR MIN_LR MIN_LR_RATIO
    LR_DECAY_STYLE LR_DECAY_FRACTION LR_DECAY_ITERS
    LR_WARMUP_FRACTION LR_WARMUP_ITERS
    LR_WSD_DECAY_FRACTION LR_WSD_DECAY_STYLE LR_WSD_DECAY_ITERS
    SAVE_INTERVAL SAVE_INTERVAL_FRACTION EVAL_INTERVAL EVAL_INTERVAL_FRACTION
    EVAL_ITERS LOG_INTERVAL WEIGHT_DECAY NUM_WORKERS PREPARE_WORKERS
    OPTIMIZER MUON_NUM_NS_STEPS MAIN_PARAMS_DTYPE MAIN_GRADS_DTYPE EXP_AVG_DTYPE EXP_AVG_SQ_DTYPE
    PACK_SAMPLES PACK_METADATA_PATH VALID_PACK_METADATA_PATH SAVE_WEIGHTS_ONLY
    FUSED_LINEAR_CROSS_ENTROPY SIMPO_BETA SIMPO_GAMMA SIMPO_LOSS_TYPE
    SIMPO_SFT_WEIGHT
)

usage() {
    cat <<'USAGE'
Usage:
  bash examples/chimera/cluster_manager.sh --config FILE COMMAND
  bash examples/chimera/cluster_manager.sh --help

Commands:
  info          Print the effective configuration and node-to-rank map.
  pull          Pull the configured image on every selected node.
  image-check   Validate mounted Chimera imports inside the image.
  preflight     Validate hosts, paths, GPUs, topology, image, and imports.
  shell         Open an interactive prepared container on the first node.
  dry-run       Print the per-node Docker launch settings without launching.
  launch        Start detached training containers on all selected nodes.
  status        Show container and GPU status on all selected nodes.
  logs          Follow shared rank-specific logs through the master node.
  docker-logs   Follow Docker logs from every selected node.
  stop          Send SIGTERM to all run containers and wait for shutdown.
  kill          Force-kill all run containers.
  cleanup       Remove stopped containers for this run.
  help          Show this help text.

Examples:
  cp examples/chimera/env/pretrain.env.example /path/to/pretrain.env
  bash examples/chimera/cluster_manager.sh --config /path/to/pretrain.env info
  bash examples/chimera/cluster_manager.sh --config /path/to/pretrain.env preflight
  bash examples/chimera/cluster_manager.sh --config /path/to/pretrain.env launch
  bash examples/chimera/cluster_manager.sh --config /path/to/pretrain.env status
  bash examples/chimera/cluster_manager.sh --config /path/to/pretrain.env logs

The manager never clones or updates repositories. Prepare the three chimera
branches under HOST_REPOS before launch. Select pretrain, context_extension,
sft, or simpo by copying the corresponding file under examples/chimera/env/.
SFT and SimPO always start a new stage from MCORE_PATH.
USAGE
}

die() {
    echo "Error: $*" >&2
    exit 1
}

q() {
    printf '%q' "$1"
}

is_true() {
    [[ "${1,,}" == true || "$1" == 1 || "${1,,}" == yes ]]
}

parse_cli() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                [[ $# -ge 2 ]] || die "--config requires a file"
                CONFIG_FILE="$2"
                shift 2
                ;;
            -h|--help|help)
                usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            -*)
                die "unknown option: $1"
                ;;
            *)
                break
                ;;
        esac
    done

    COMMAND="${1:-}"
    [[ -n "$COMMAND" ]] || { usage; exit 2; }
    [[ $# -le 1 ]] || die "unexpected argument: $2"
}

load_config() {
    [[ -n "$CONFIG_FILE" ]] || die "--config FILE is required"
    [[ -f "$CONFIG_FILE" ]] || die "configuration file not found: $CONFIG_FILE"

    # Configuration files are trusted shell assignments maintained by the operator.
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"

    STAGE="${STAGE:-pretrain}"
    RUN_NAME="${RUN_NAME:-}"
    NODES_CSV="${NODES_CSV:-dgx11,dgx12,dgx13,dgx14,dgx15,dgx16}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    MASTER_ADDR="${MASTER_ADDR:-}"
    MASTER_PORT="${MASTER_PORT:-29500}"

    CHIMERA_ROOT="${CHIMERA_ROOT:-/nvme_zone3/home/ekamai1/chimera}"
    HOST_REPOS="${HOST_REPOS:-$CHIMERA_ROOT/repos}"
    HOST_DATA="${HOST_DATA:-$CHIMERA_ROOT/data}"
    CONTAINER_REPOS="${CONTAINER_REPOS:-/workspace/repos}"
    CONTAINER_DATA="${CONTAINER_DATA:-/datasets/megadata}"
    IMAGE="${IMAGE:-suryavikram6/megatron-gemma:v2-fixed}"

    TOKENIZER_MODEL="${TOKENIZER_MODEL:-$CONTAINER_DATA/hf_models/chimera-10b}"
    RUNS_ROOT="${RUNS_ROOT:-$CONTAINER_DATA/runs/$STAGE}"
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$CONTAINER_DATA/pretrain/train_text_document}"
    VALID_DATA_PATH="${VALID_DATA_PATH-}"
    LOAD_CHECKPOINT="${LOAD_CHECKPOINT-}"
    DATA_PATH="${DATA_PATH:-$CONTAINER_DATA/$STAGE/train.jsonl}"
    MCORE_PATH="${MCORE_PATH:-$CONTAINER_DATA/checkpoints/pretrain}"

    TP_SIZE="${TP_SIZE:-1}"
    PP_SIZE="${PP_SIZE:-1}"
    EP_SIZE="${EP_SIZE:-1}"
    CP_SIZE="${CP_SIZE:-1}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"

    INTRA_DOC_MASKING="${INTRA_DOC_MASKING:-false}"
    CONTEXT_PHASE="${CONTEXT_PHASE:-8k}"
    SEQ_LENGTH="${SEQ_LENGTH:-8192}"
    TRAIN_TOKENS="${TRAIN_TOKENS-}"
    TRAIN_ITERS="${TRAIN_ITERS-}"
    TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
    LR="${LR:-3e-4}"
    MIN_LR="${MIN_LR-}"
    MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"
    LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
    LR_DECAY_FRACTION="${LR_DECAY_FRACTION:-1.0}"
    LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.005}"
    LR_WSD_DECAY_FRACTION="${LR_WSD_DECAY_FRACTION:-0.10}"
    LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-linear}"
    SAVE_INTERVAL="${SAVE_INTERVAL-}"
    SAVE_INTERVAL_FRACTION="${SAVE_INTERVAL_FRACTION:-0.025}"
    EVAL_INTERVAL="${EVAL_INTERVAL-}"
    EVAL_INTERVAL_FRACTION="${EVAL_INTERVAL_FRACTION:-0.005}"
    EVAL_ITERS="${EVAL_ITERS:-4}"
    LOG_INTERVAL="${LOG_INTERVAL:-1}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
    NUM_WORKERS="${NUM_WORKERS:-32}"
    PREPARE_WORKERS="${PREPARE_WORKERS:-32}"
    OPTIMIZER="${OPTIMIZER:-adam}"
    MUON_NUM_NS_STEPS="${MUON_NUM_NS_STEPS:-6}"
    MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"
    MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
    EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
    EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"
    PACK_SAMPLES="${PACK_SAMPLES:-true}"
    PACK_METADATA_PATH="${PACK_METADATA_PATH-}"
    VALID_PACK_METADATA_PATH="${VALID_PACK_METADATA_PATH-}"
    SAVE_WEIGHTS_ONLY="${SAVE_WEIGHTS_ONLY:-false}"
    FUSED_LINEAR_CROSS_ENTROPY="${FUSED_LINEAR_CROSS_ENTROPY:-true}"
    SIMPO_BETA="${SIMPO_BETA:-2.5}"
    SIMPO_GAMMA="${SIMPO_GAMMA:-0.55}"
    SIMPO_LOSS_TYPE="${SIMPO_LOSS_TYPE:-sigmoid}"
    SIMPO_SFT_WEIGHT="${SIMPO_SFT_WEIGHT:-0.0}"

    ENABLE_GPU="${ENABLE_GPU:-true}"
    ENABLE_IB="${ENABLE_IB:-true}"
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
    NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
    NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-false}"
    FORCE="${FORCE:-false}"
    SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-false}"
    STOP_TIMEOUT="${STOP_TIMEOUT:-1800}"
    SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
    TAIL_LINES="${TAIL_LINES:-200}"

    IFS=',' read -r -a NODES <<< "$NODES_CSV"
    NNODES=${#NODES[@]}
    (( NNODES > 0 )) || die "NODES_CSV must contain at least one node"
    MASTER_NODE="${NODES[0]}"
    WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

    [[ "$STAGE" =~ ^(pretrain|context_extension|sft|simpo)$ ]] || die "STAGE must be pretrain, context_extension, sft, or simpo"
    [[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "RUN_NAME must use letters, numbers, dot, underscore, or dash"
    [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]] || die "GPUS_PER_NODE must be a non-negative integer"
    [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be an integer"

    if [[ "$STAGE" == pretrain || "$STAGE" == context_extension ]]; then
        if [[ -z "$TRAIN_TOKENS" ]]; then
            if [[ "$STAGE" == pretrain ]]; then
                TRAIN_TOKENS=440000000000
            else
                TRAIN_TOKENS=10000000000
            fi
        fi
        TOKENS_PER_ITER=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
        if [[ -z "$TRAIN_ITERS" ]]; then
            TRAIN_ITERS=$(chimera_ceil_div "$TRAIN_TOKENS" "$TOKENS_PER_ITER")
        fi
        chimera_resolve_schedule
    fi

    if [[ -z "$MASTER_ADDR" ]]; then
        if (( NNODES == 1 )); then
            MASTER_ADDR=127.0.0.1
        else
            MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" 2>/dev/null | awk 'NR == 1 {print $1}' || true)
            [[ -n "$MASTER_ADDR" ]] || die "cannot resolve MASTER_NODE=$MASTER_NODE; set MASTER_ADDR explicitly"
        fi
    fi

    CONTAINER_RUN_DIR="$RUNS_ROOT/$RUN_NAME"
    HOST_RUN_DIR=$(container_to_host "$CONTAINER_RUN_DIR")
    CONTAINER_PREFIX="chimera_${STAGE}_${RUN_NAME//[^A-Za-z0-9_.-]/_}"
}

container_to_host() {
    local path="$1"
    if [[ "$path" == "$CONTAINER_DATA" ]]; then
        printf '%s\n' "$HOST_DATA"
    elif [[ "$path" == "$CONTAINER_DATA/"* ]]; then
        printf '%s/%s\n' "$HOST_DATA" "${path#"$CONTAINER_DATA/"}"
    elif [[ "$path" == "$CONTAINER_REPOS" ]]; then
        printf '%s\n' "$HOST_REPOS"
    elif [[ "$path" == "$CONTAINER_REPOS/"* ]]; then
        printf '%s/%s\n' "$HOST_REPOS" "${path#"$CONTAINER_REPOS/"}"
    else
        die "container path is outside configured mounts: $path"
    fi
}

container_name_for_rank() {
    printf '%s_rank%s\n' "$CONTAINER_PREFIX" "$1"
}

is_local_node() {
    local node="$1"
    local short_host full_host
    short_host=$(hostname -s)
    full_host=$(hostname -f 2>/dev/null || hostname)
    [[ "$node" == localhost || "$node" == 127.0.0.1 || "$node" == "$short_host" || "$node" == "$full_host" ]]
}

run_node() {
    local node="$1"
    shift
    if is_local_node "$node"; then
        "$@"
    else
        local remote_command
        printf -v remote_command '%q ' "$@"
        ssh -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
            "$node" "${remote_command% }"
    fi
}

run_node_script() {
    local node="$1"
    local script="$2"
    shift 2

    local assignment name value quoted payload=""
    for assignment in "$@"; do
        [[ "$assignment" == *=* ]] || die "invalid remote assignment: $assignment"
        name=${assignment%%=*}
        value=${assignment#*=}
        [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "invalid remote variable name: $name"
        printf -v quoted '%q' "$value"
        payload+="export $name=$quoted"$'\n'
    done

    payload+="$script"$'\n'
    run_node "$node" bash -s <<< "$payload"
}

validate_topology() {
    local value
    for value in TP_SIZE PP_SIZE EP_SIZE CP_SIZE MICRO_BATCH_SIZE; do
        [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || die "$value must be a positive integer"
    done
    if ! is_true "$ENABLE_GPU"; then
        return
    fi

    (( GPUS_PER_NODE > 0 )) || die "GPUS_PER_NODE must be positive when ENABLE_GPU=true"
    local model_parallel=$((TP_SIZE * PP_SIZE * CP_SIZE))
    (( WORLD_SIZE % model_parallel == 0 )) || die "WORLD_SIZE=$WORLD_SIZE must be divisible by TP*PP*CP=$model_parallel"
    (( 32 % EP_SIZE == 0 )) || die "EP_SIZE=$EP_SIZE must divide Chimera's 32 experts"
    (( WORLD_SIZE % (EP_SIZE * PP_SIZE) == 0 )) || die "WORLD_SIZE=$WORLD_SIZE must be divisible by EP*PP=$((EP_SIZE * PP_SIZE)) because expert TP is fixed at 1"
    [[ "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "GLOBAL_BATCH_SIZE must be a positive integer"
    local dp_size=$((WORLD_SIZE / model_parallel))
    local batch_unit=$((MICRO_BATCH_SIZE * dp_size))
    (( GLOBAL_BATCH_SIZE >= batch_unit )) || die "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is smaller than microbatch*DP=$batch_unit"
    (( GLOBAL_BATCH_SIZE % batch_unit == 0 )) || die "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by microbatch*DP=$batch_unit"
}

print_info() {
    validate_topology
    local dp_size="disabled"
    if is_true "$ENABLE_GPU"; then
        dp_size=$((WORLD_SIZE / (TP_SIZE * PP_SIZE * CP_SIZE)))
    fi

    cat <<EOF
============================================================
Chimera cluster run
Config:             $CONFIG_FILE
Stage:              $STAGE
Run name:           $RUN_NAME
Nodes:              ${NODES[*]}
NNODES:             $NNODES
GPUs per node:      $GPUS_PER_NODE
World size:         $WORLD_SIZE
Master:             $MASTER_NODE ($MASTER_ADDR:$MASTER_PORT)
Image:              $IMAGE
Host repos:         $HOST_REPOS
Host data:          $HOST_DATA
Container run dir:  $CONTAINER_RUN_DIR
Host run dir:       $HOST_RUN_DIR
Parallelism:        TP=$TP_SIZE PP=$PP_SIZE EP=$EP_SIZE CP=$CP_SIZE DP=$dp_size
Batch:              micro=$MICRO_BATCH_SIZE global=${GLOBAL_BATCH_SIZE:-unset}
Context:            phase=$CONTEXT_PHASE sequence=$SEQ_LENGTH
Duration:           tokens=${TRAIN_TOKENS:-unset} iters=${TRAIN_ITERS:-packing-derived} epochs=$TRAIN_EPOCHS
Validation:         ${VALID_DATA_PATH:-disabled}
LR schedule:        style=$LR_DECAY_STYLE lr=$LR min=${MIN_LR:-ratio:$MIN_LR_RATIO} warmup=${LR_WARMUP_ITERS:-fraction:$LR_WARMUP_FRACTION}
Intervals:          log=$LOG_INTERVAL save=${SAVE_INTERVAL:-fraction:$SAVE_INTERVAL_FRACTION} eval=${EVAL_INTERVAL:-fraction:$EVAL_INTERVAL_FRACTION}x$EVAL_ITERS
Optimizer:          $OPTIMIZER (params/grads/moments: $MAIN_PARAMS_DTYPE/$MAIN_GRADS_DTYPE/$EXP_AVG_DTYPE,$EXP_AVG_SQ_DTYPE)
GPU enabled:        $ENABLE_GPU
InfiniBand enabled: $ENABLE_IB
============================================================
EOF
    local rank
    for rank in "${!NODES[@]}"; do
        printf 'rank=%s node=%s container=%s\n' "$rank" "${NODES[$rank]}" "$(container_name_for_rank "$rank")"
    done
}

stage_host_paths() {
    local paths=(
        "$HOST_REPOS/Megatron-LM"
        "$HOST_REPOS/Megatron-Bridge"
        "$HOST_REPOS/transformers"
        "$HOST_DATA"
        "$(container_to_host "$TOKENIZER_MODEL")"
    )

    case "$STAGE" in
        pretrain|context_extension)
            paths+=("$(container_to_host "$TRAIN_DATA_PATH").bin" "$(container_to_host "$TRAIN_DATA_PATH").idx")
            if [[ -n "$VALID_DATA_PATH" ]]; then
                paths+=("$(container_to_host "$VALID_DATA_PATH").bin" "$(container_to_host "$VALID_DATA_PATH").idx")
            fi
            if [[ -n "$LOAD_CHECKPOINT" ]]; then
                paths+=("$(container_to_host "$LOAD_CHECKPOINT")/latest_checkpointed_iteration.txt")
            fi
            ;;
        sft|simpo)
            paths+=("$(container_to_host "$DATA_PATH")" "$(container_to_host "$MCORE_PATH")")
            if [[ -n "$VALID_DATA_PATH" ]]; then
                paths+=("$(container_to_host "$VALID_DATA_PATH")")
            fi
            ;;
    esac

    printf '%s\n' "${paths[@]}"
}

image_check_node() {
    local node="$1"
    local check_script
    read -r -d '' check_script <<'CHECK' || true
set -euo pipefail
export PATH="/workspace/venv/bin:$PATH"
python3 - <<'PY'
import transformers
from transformers import AutoTokenizer, ChimeraConfig, ChimeraForCausalLM
import megatron.bridge

import os

repos = os.environ["CONTAINER_REPOS"]
tokenizer_path = f"{repos}/transformers/src/transformers/models/chimera/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
config = ChimeraConfig()
assert (config.first_k_dense_replace, config.last_k_dense_replace) == (2, 0)
assert len(tokenizer) == 50176
assert tokenizer.convert_tokens_to_ids("<start_of_turn>") == 2
assert tokenizer.convert_tokens_to_ids("<end_of_turn>") == 3
assert tokenizer.unk_token is None
print("chimera_image_ok", transformers.__file__, megatron.bridge.__file__)
PY
for script in train.sh context_extend.sh sft.sh simpo.sh schedule_helpers.sh; do
    bash -n "$CONTAINER_REPOS/Megatron-LM/examples/chimera/$script"
done
CHECK

    local remote_script
    read -r -d '' remote_script <<'REMOTE' || true
set -euo pipefail
args=(
    run --rm
    --network host
    --ipc host
    --mount "type=bind,src=$HOST_REPOS,dst=$CONTAINER_REPOS"
    --mount "type=bind,src=$HOST_DATA,dst=$CONTAINER_DATA"
    --env "VIRTUAL_ENV=/workspace/venv"
    --env "HF_HOME=$CONTAINER_DATA/cache/huggingface"
    --env "CONTAINER_REPOS=$CONTAINER_REPOS"
    --env "PYTHONPATH=$CONTAINER_REPOS/Megatron-LM:$CONTAINER_REPOS/Megatron-Bridge/src:$CONTAINER_REPOS/transformers/src"
    --workdir "$CONTAINER_REPOS/Megatron-LM"
)
if [[ "$ENABLE_GPU" == true ]]; then
    args+=(--gpus all)
fi
args+=("$IMAGE" bash -lc "$CHECK_SCRIPT")
docker "${args[@]}"
REMOTE

    run_node_script "$node" "$remote_script" \
        "HOST_REPOS=$HOST_REPOS" \
        "HOST_DATA=$HOST_DATA" \
        "CONTAINER_REPOS=$CONTAINER_REPOS" \
        "CONTAINER_DATA=$CONTAINER_DATA" \
        "ENABLE_GPU=$ENABLE_GPU" \
        "IMAGE=$IMAGE" \
        "CHECK_SCRIPT=$check_script"
}

preflight_node() {
    local node="$1"
    local rank="$2"
    local container_name
    container_name=$(container_name_for_rank "$rank")
    local required_paths
    required_paths=$(stage_host_paths)

    echo "==================== PREFLIGHT $node rank=$rank ===================="
    local remote_script
    read -r -d '' remote_script <<'REMOTE' || true
set -euo pipefail
fail=0
echo "host=$(hostname) date=$(date -Is)"

command -v docker >/dev/null 2>&1 || { echo "FAIL: docker is missing"; fail=1; }
if command -v docker >/dev/null 2>&1; then
    docker info >/dev/null 2>&1 || { echo "FAIL: Docker daemon is unavailable"; fail=1; }
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FAIL: image is missing: $IMAGE"; fail=1; }
    if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
        echo "FAIL: container already exists: $CONTAINER_NAME"
        fail=1
    fi
fi

while IFS= read -r path; do
    [[ -e "$path" ]] || { echo "FAIL: missing path: $path"; fail=1; }
done <<< "$REQUIRED_PATHS"

[[ -d "$HOST_DATA" && -w "$HOST_DATA" ]] || { echo "FAIL: data root is not writable: $HOST_DATA"; fail=1; }

bridge_link="$HOST_REPOS/Megatron-Bridge/3rdparty/Megatron-LM"
if [[ -e "$bridge_link" && ! -L "$bridge_link" ]]; then
    echo "FAIL: Bridge Megatron-LM path exists but is not a symlink"
    fail=1
else
    ln -sfn ../../Megatron-LM "$bridge_link" || {
        echo "FAIL: could not create Bridge Megatron-LM symlink"
        fail=1
    }
fi

if [[ "$ENABLE_GPU" == true ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || { echo "FAIL: nvidia-smi is missing"; fail=1; }
    if command -v nvidia-smi >/dev/null 2>&1; then
        actual=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        [[ "$actual" == "$GPUS_PER_NODE" ]] || { echo "FAIL: expected $GPUS_PER_NODE GPUs, found $actual"; fail=1; }
        apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
        if [[ -n "$apps" && "$ALLOW_BUSY_GPUS" != true ]]; then
            echo "FAIL: GPU compute processes are already running"
            fail=1
        fi
    fi
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia || { echo "FAIL: NVIDIA Docker runtime is unavailable"; fail=1; }
fi

if [[ "$ENABLE_IB" == true && ! -d /dev/infiniband ]]; then
    echo "FAIL: /dev/infiniband is missing"
    fail=1
fi

if [[ "$NODE_RANK" == 0 ]] && command -v ss >/dev/null 2>&1; then
    if ss -lnt | awk '{print $4}' | grep -Eq "(^|:)$MASTER_PORT$"; then
        echo "FAIL: master port is already in use: $MASTER_PORT"
        fail=1
    fi
fi

df -h "$HOST_DATA" /var/lib/docker 2>/dev/null || true
[[ "$fail" == 0 ]] || exit 1
echo "PREFLIGHT_RESULT=PASS"
REMOTE

    run_node_script "$node" "$remote_script" \
        "IMAGE=$IMAGE" \
        "CONTAINER_NAME=$container_name" \
        "REQUIRED_PATHS=$required_paths" \
        "HOST_REPOS=$HOST_REPOS" \
        "HOST_DATA=$HOST_DATA" \
        "ENABLE_GPU=$ENABLE_GPU" \
        "ENABLE_IB=$ENABLE_IB" \
        "GPUS_PER_NODE=$GPUS_PER_NODE" \
        "ALLOW_BUSY_GPUS=$ALLOW_BUSY_GPUS" \
        "NODE_RANK=$rank" \
        "MASTER_PORT=$MASTER_PORT"
}

cmd_preflight() {
    print_info
    [[ ! -e "$HOST_RUN_DIR" ]] || die "run directory already exists: $HOST_RUN_DIR"

    local pids=()
    local rank
    for rank in "${!NODES[@]}"; do
        preflight_node "${NODES[$rank]}" "$rank" &
        pids+=("$!")
    done
    local rc=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    (( rc == 0 )) || die "preflight failed on one or more nodes"

    echo "Running container import checks..."
    pids=()
    for node in "${NODES[@]}"; do
        image_check_node "$node" &
        pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    (( rc == 0 )) || die "image check failed on one or more nodes"

    local expected=""
    for node in "${NODES[@]}"; do
        local image_id
        image_id=$(run_node "$node" docker image inspect "$IMAGE" --format '{{.Id}}')
        if [[ -z "$expected" ]]; then
            expected="$image_id"
        elif [[ "$image_id" != "$expected" ]]; then
            die "image ID differs on $node: $image_id != $expected"
        fi
    done
    echo "All nodes use image $expected"
}

cmd_pull() {
    local pids=() node
    for node in "${NODES[@]}"; do
        (echo "Pulling $IMAGE on $node"; run_node "$node" docker pull "$IMAGE") &
        pids+=("$!")
    done
    local rc=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    (( rc == 0 )) || die "image pull failed on one or more nodes"
}

cmd_image_check() {
    local pids=() node
    for node in "${NODES[@]}"; do
        image_check_node "$node" &
        pids+=("$!")
    done
    local rc=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    (( rc == 0 )) || die "image check failed on one or more nodes"
}

prepare_run_metadata() {
    local commits node_map
    commits=$(for repo in Megatron-LM Megatron-Bridge transformers; do
        if git -C "$HOST_REPOS/$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            git -C "$HOST_REPOS/$repo" log -1 --format="$repo %H %s"
        else
            echo "$repo not-a-git-checkout"
        fi
    done)
    node_map=$(for rank in "${!NODES[@]}"; do printf '%s %s\n' "$rank" "${NODES[$rank]}"; done)

    mkdir -p "$HOST_RUN_DIR/logs"
    cp "$CONFIG_FILE" "$HOST_RUN_DIR/cluster.env"
    printf '%s\n' "$commits" > "$HOST_RUN_DIR/git_commits.txt"
    printf '%s\n' "$node_map" > "$HOST_RUN_DIR/node_map.txt"
    run_node "$MASTER_NODE" docker image inspect "$IMAGE" --format '{{.Id}} {{json .RepoDigests}}' > "$HOST_RUN_DIR/image.txt"
    print_info > "$HOST_RUN_DIR/effective_config.txt"
}

launch_node() {
    local node="$1"
    local rank="$2"
    local dry_run="$3"
    local container_name
    container_name=$(container_name_for_rank "$rank")

    if is_true "$dry_run"; then
        cat <<EOF
node=$node rank=$rank container=$container_name
  image=$IMAGE
  mounts=$HOST_REPOS:$CONTAINER_REPOS,$HOST_DATA:$CONTAINER_DATA
  distributed=NNODES=$NNODES NODE_RANK=$rank GPUS_PER_NODE=$GPUS_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
  stage=$STAGE run=$RUN_NAME
EOF
        return
    fi

    local remote_script
    read -r -d '' remote_script <<'REMOTE' || true
set -euo pipefail

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    if [[ "$FORCE" == true ]]; then
        docker rm -f "$CONTAINER_NAME" >/dev/null
    else
        echo "Container already exists: $CONTAINER_NAME" >&2
        exit 20
    fi
fi

args=(
    run -d
    --name "$CONTAINER_NAME"
    --hostname "$NODE_NAME"
    --label "chimera.run=$RUN_NAME"
    --label "chimera.stage=$STAGE"
    --label "chimera.node=$NODE_NAME"
    --label "chimera.rank=$NODE_RANK"
    --network host
    --ipc host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --ulimit nofile=1048576:1048576
    --log-driver json-file
    --log-opt max-size=200m
    --log-opt max-file=5
    --mount "type=bind,src=$HOST_REPOS,dst=$CONTAINER_REPOS"
    --mount "type=bind,src=$HOST_DATA,dst=$CONTAINER_DATA"
    --env "VIRTUAL_ENV=/workspace/venv"
    --env "HF_HOME=$CONTAINER_DATA/cache/huggingface"
    --env "PYTHONUNBUFFERED=1"
    --env "PYTHONPATH=$CONTAINER_REPOS/Megatron-LM:$CONTAINER_REPOS/Megatron-Bridge/src:$CONTAINER_REPOS/transformers/src"
    --env "STAGE=$STAGE"
    --env "RUN_STAMP=$RUN_NAME"
    --env "NNODES=$NNODES"
    --env "NODE_RANK=$NODE_RANK"
    --env "GPUS_PER_NODE=$GPUS_PER_NODE"
    --env "MASTER_ADDR=$MASTER_ADDR"
    --env "MASTER_PORT=$MASTER_PORT"
    --env "NCCL_DEBUG=$NCCL_DEBUG"
    --workdir "$CONTAINER_REPOS/Megatron-LM"
)
for env_name in $FORWARDED_ENV_NAMES; do
    args+=(--env "$env_name=${!env_name-}")
done

if [[ "$ENABLE_GPU" == true ]]; then
    args+=(--gpus all)
fi
if [[ "$ENABLE_IB" == true ]]; then
    args+=(--device=/dev/infiniband)
    args+=(--env "NCCL_IB_DISABLE=0" --env "NCCL_IB_HCA=$NCCL_IB_HCA")
else
    args+=(--env "NCCL_IB_DISABLE=1")
fi
if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    args+=(--env "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
fi

read -r -d '' container_command <<'COMMAND' || true
set -euo pipefail
export PATH="/workspace/venv/bin:$PATH"
case "$STAGE" in
    pretrain) script=examples/chimera/train.sh ;;
    context_extension) script=examples/chimera/context_extend.sh ;;
    sft) script=examples/chimera/sft.sh ;;
    simpo) script=examples/chimera/simpo.sh ;;
    *) echo "Invalid STAGE=$STAGE" >&2; exit 2 ;;
esac
exec bash "$script"
COMMAND

args+=("$IMAGE" bash -lc "$container_command")
docker "${args[@]}"
REMOTE

    local forwarded_env_names="${FORWARDED_ENV_VARS[*]}"
    local remote_assignments=(
        "CONTAINER_NAME=$container_name" \
        "NODE_NAME=$node" \
        "NODE_RANK=$rank" \
        "FORCE=$FORCE" \
        "HOST_REPOS=$HOST_REPOS" \
        "HOST_DATA=$HOST_DATA" \
        "CONTAINER_REPOS=$CONTAINER_REPOS" \
        "CONTAINER_DATA=$CONTAINER_DATA" \
        "STAGE=$STAGE" \
        "RUN_NAME=$RUN_NAME" \
        "NNODES=$NNODES" \
        "GPUS_PER_NODE=$GPUS_PER_NODE" \
        "MASTER_ADDR=$MASTER_ADDR" \
        "MASTER_PORT=$MASTER_PORT" \
        "ENABLE_GPU=$ENABLE_GPU" \
        "ENABLE_IB=$ENABLE_IB" \
        "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" \
        "NCCL_IB_HCA=$NCCL_IB_HCA" \
        "NCCL_DEBUG=$NCCL_DEBUG" \
        "IMAGE=$IMAGE" \
        "FORWARDED_ENV_NAMES=$forwarded_env_names"
    )
    local env_name
    for env_name in "${FORWARDED_ENV_VARS[@]}"; do
        remote_assignments+=("$env_name=${!env_name-}")
    done
    run_node_script "$node" "$remote_script" "${remote_assignments[@]}"
}

cmd_dry_run() {
    print_info
    local rank
    for rank in "${!NODES[@]}"; do
        launch_node "${NODES[$rank]}" "$rank" true
    done
}

cmd_launch() {
    validate_topology
    is_true "$ENABLE_GPU" || die "launch requires ENABLE_GPU=true; use image-check, preflight, or dry-run on CPU hosts"
    [[ ! -e "$HOST_RUN_DIR" ]] || die "run directory already exists: $HOST_RUN_DIR"
    if ! is_true "$SKIP_PREFLIGHT"; then
        cmd_preflight
    fi
    prepare_run_metadata

    local pids=() rank
    for rank in "${!NODES[@]}"; do
        launch_node "${NODES[$rank]}" "$rank" false &
        pids+=("$!")
    done
    local rc=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    if (( rc != 0 )); then
        echo "A node failed to launch; stopping containers from this run." >&2
        cmd_stop || true
        exit 1
    fi

    sleep 15
    cmd_status

    local startup_failed=0 state
    for rank in "${!NODES[@]}"; do
        state=$(run_node "${NODES[$rank]}" docker inspect \
            --format '{{.State.Status}} {{.State.ExitCode}}' \
            "$(container_name_for_rank "$rank")" 2>/dev/null || true)
        if [[ "$state" != "running 0" && "$state" != "exited 0" ]]; then
            echo "Startup failed on ${NODES[$rank]} rank=$rank: ${state:-container missing}" >&2
            run_node "${NODES[$rank]}" docker logs --tail 100 \
                "$(container_name_for_rank "$rank")" >&2 2>/dev/null || true
            startup_failed=1
        fi
    done
    if (( startup_failed != 0 )); then
        cmd_stop || true
        die "one or more training containers failed during startup"
    fi
}

cmd_status() {
    local pids=() rank
    for rank in "${!NODES[@]}"; do
        local node="${NODES[$rank]}"
        local name
        name=$(container_name_for_rank "$rank")
        (
            echo "==================== STATUS $node rank=$rank ===================="
            run_node "$node" bash -lc "docker ps -a --filter name=^/$(q "$name")\$ --format 'table {{.Names}}\\t{{.Status}}\\t{{.Image}}'; docker inspect $(q "$name") --format 'status={{.State.Status}} exit_code={{.State.ExitCode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' 2>/dev/null || true; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true; fi"
        ) &
        pids+=("$!")
    done
    local rc=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    return "$rc"
}

cmd_logs() {
    echo "Following $HOST_RUN_DIR/logs on $MASTER_NODE; Ctrl-C stops tailing only."
    run_node "$MASTER_NODE" bash -lc "shopt -s nullglob; files=($(q "$HOST_RUN_DIR")/logs/*.log); (( \${#files[@]} > 0 )) || { echo 'No shared logs found yet'; exit 1; }; tail -n $(q "$TAIL_LINES") -F \"\${files[@]}\""
}

cmd_docker_logs() {
    local pids=() rank
    for rank in "${!NODES[@]}"; do
        local node="${NODES[$rank]}"
        local name
        name=$(container_name_for_rank "$rank")
        (run_node "$node" docker logs --tail "$TAIL_LINES" -f "$name" 2>&1 | sed -u "s/^/[$node rank$rank] /") &
        pids+=("$!")
    done
    wait "${pids[@]}"
}

cmd_stop() {
    local pids=() rank
    for rank in "${!NODES[@]}"; do
        local node="${NODES[$rank]}"
        local name
        name=$(container_name_for_rank "$rank")
        (run_node "$node" docker stop -t "$STOP_TIMEOUT" "$name" 2>/dev/null || true) &
        pids+=("$!")
    done
    wait "${pids[@]}"
}

cmd_kill() {
    local pids=() rank
    for rank in "${!NODES[@]}"; do
        local node="${NODES[$rank]}"
        local name
        name=$(container_name_for_rank "$rank")
        (run_node "$node" docker kill "$name" 2>/dev/null || true) &
        pids+=("$!")
    done
    wait "${pids[@]}"
}

cmd_cleanup() {
    local pids=() rank
    for rank in "${!NODES[@]}"; do
        local node="${NODES[$rank]}"
        local name
        name=$(container_name_for_rank "$rank")
        (run_node "$node" docker rm "$name" 2>/dev/null || true) &
        pids+=("$!")
    done
    wait "${pids[@]}"
}

cmd_shell() {
    local node="$MASTER_NODE"
    local gpu_args=()
    local ib_args=()
    if is_true "$ENABLE_GPU"; then gpu_args=(--gpus all); fi
    if is_true "$ENABLE_IB"; then ib_args=(--device=/dev/infiniband); fi

    local docker_command
    printf -v docker_command '%q ' docker run --rm -it \
        "${gpu_args[@]}" "${ib_args[@]}" \
        --network host --ipc host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --mount "type=bind,src=$HOST_REPOS,dst=$CONTAINER_REPOS" \
        --mount "type=bind,src=$HOST_DATA,dst=$CONTAINER_DATA" \
        --env "VIRTUAL_ENV=/workspace/venv" \
        --env "HF_HOME=$CONTAINER_DATA/cache/huggingface" \
        --env "PYTHONPATH=$CONTAINER_REPOS/Megatron-LM:$CONTAINER_REPOS/Megatron-Bridge/src:$CONTAINER_REPOS/transformers/src" \
        --workdir "$CONTAINER_REPOS/Megatron-LM" \
        "$IMAGE" bash -lc 'export PATH="/workspace/venv/bin:$PATH"; exec bash --noprofile --norc'

    if is_local_node "$node"; then
        eval "$docker_command"
    else
        ssh -t -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" "$node" "$docker_command"
    fi
}

parse_cli "$@"
load_config

case "$COMMAND" in
    info) print_info ;;
    pull) cmd_pull ;;
    image-check) cmd_image_check ;;
    preflight) cmd_preflight ;;
    shell) cmd_shell ;;
    dry-run) cmd_dry_run ;;
    launch) cmd_launch ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    docker-logs) cmd_docker_logs ;;
    stop) cmd_stop ;;
    kill) cmd_kill ;;
    cleanup) cmd_cleanup ;;
    help) usage ;;
    *) die "unknown command: $COMMAND" ;;
esac
