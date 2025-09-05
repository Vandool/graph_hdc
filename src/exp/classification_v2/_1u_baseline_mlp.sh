#!/usr/bin/env bash
# universal_cluster_submit.sh
# Submit the same experiment to multiple partitions depending on $CLUSTER.
# Supported CLUSTER values:
#   uc3  -> bwUniCluster 3.0
#       Available Nodes : cpu_il dev_cpu_il | cpu dev_cpu | highmem dev_highmem | gpu_h100 dev_gpu_h100 | gpu_mi300 | gpu_a100_il gpu_h100_il|
#   hk   -> HoreKa
#       Available Nodes: # Available partitions: cpuonly large accelerated accelerated-h100 accelerated-200
#                        # Available dev partitions: dev_cpuonly dev_accelerated dev_accelerated-h100
#   else -> HAICORE (default)
# Usage for dry-run:
# ```sh
# CLUSTER=uc3 DRY_RUN=1 ./universal_cluster_submit.sh
# ```
# Usage for targeting partitions:
# ```sh
# CLUSTER=uc3 ONLY_PARTITIONS=gpu_h100,gpu_a100_il ./universal_cluster_submit.sh
# ```

set -euo pipefail

# -----------------------------
# User-configurable parameters
# -----------------------------

# Cluster selector: uc3 | hk | (anything else = haicore)
CLUSTER="${CLUSTER:-uc3}"

# Slurm common settings
JOB_NAME="${JOB_NAME:-MLP_stratified_base}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"

# Module loading (override per cluster below if needed)
MODULE_LOAD_DEFAULT='module load devel/cuda'

# Paths
PROJECT_DIR="${GHDC_HOME}"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/classification_v2"
SCRIPT_NAME="${SCRIPT_NAME:-1_mlp.py}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# W&B (optional)
ENTITY="${ENTITY:-akaveh}"
PROJECT="${PROJECT:-graph_hdc}"
EXP_NAME="${EXP_NAME:-mlp_stratified_base_batchnorm_lr5e-4}"

# Dry run to preview sbatch commands without submitting: 0 or 1
DRY_RUN="${DRY_RUN:-0}"

# Optional partition filter (comma-separated). If set, only those partitions are submitted.
ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"

# -----------------------------
# Python args (edit as needed)
# -----------------------------
# Aligned with classification_v2/1_mlp.py parameters in your example.
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HV_DIM="${HV_DIM:-7744}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MICRO_BS="${MICRO_BS:-64}"
SAVE_EVERY_SECONDS="${SAVE_EVERY_SECONDS:-3600}"
KEEP_LAST_K="${KEEP_LAST_K:-2}"
STRATIFY="${STRATIFY:-True}"
P_PER_PARENT="${P_PER_PARENT:-20}"
N_PER_PARENT="${N_PER_PARENT:-20}"
ORACLE_BEAM_SIZE="${ORACLE_BEAM_SIZE:-8}"
ORACLE_NUM_EVALS="${ORACLE_NUM_EVALS:-4}"
RESAMPLE_TRAINING_DATA_ON_BATCH="${RESAMPLE_TRAINING_DATA_ON_BATCH:-True}"
USE_BATCH_NORM="${USE_BATCH_NORM:-True}"
USE_LAYER_NORM="${USE_LAYER_NORM:-False}"

# Build python args array
PY_ARGS=(
  "$SCRIPT"
  --project_dir "$PROJECT_DIR"
  --exp_dir_name "$EXP_NAME"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --hv_dim "$HV_DIM"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --num_workers "$NUM_WORKERS"
  --micro_bs "$MICRO_BS"
  --save_every_seconds "$SAVE_EVERY_SECONDS"
  --keep_last_k "$KEEP_LAST_K"
  --stratify "$STRATIFY"
  --p_per_parent "$P_PER_PARENT"
  --n_per_parent "$N_PER_PARENT"
  --oracle_beam_size "$ORACLE_BEAM_SIZE"
  --oracle_num_evals "$ORACLE_NUM_EVALS"
  --resample_training_data_on_batch "$RESAMPLE_TRAINING_DATA_ON_BATCH"
  --use_batch_norm "$USE_BATCH_NORM"
  --use_layer_norm "$USE_LAYER_NORM"
)

# ---------------------------------
# Partition/time/mem tuples by site
# ---------------------------------
# NOTE: comments indicate stated max walltimes for reference.

case "$CLUSTER" in
  uc3)
    MODULE_LOAD="$MODULE_LOAD_DEFAULT"
    # Partitions (bwUniCluster 3.0 — GPU dev/standard)
    # cpu_il dev_cpu_il | cpu dev_cpu | highmem dev_highmem | gpu_h100 dev_gpu_h100 | gpu_mi300 | gpu_a100_il gpu_h100_il
    TUPLES=$(
      cat <<'EOF'
gpu_h100|36:00:00|64G
gpu_a100_il|36:00:00|64G
gpu_h100_il|36:00:00|64G
EOF
    )
    ;;
  hk)
    # HoreKa
    MODULE_LOAD="$MODULE_LOAD_DEFAULT"
    # (accelerated, 24:00:00, 64G)   # max 48:00:00
    # (accelerated-h100, 24:00:00, 64G) # max 48:00:00
    TUPLES=$(
      cat <<'EOF'
accelerated|36:00:00|64G
accelerated-h100|36:00:00|64G
EOF
    )
    ;;
  *)
    # HAICORE default
    MODULE_LOAD="$MODULE_LOAD_DEFAULT"
    # (normal, 24:00:00, 64G)   # max 72:00:00
    # (advanced, 24:00:00, 64G) # max 72:00:00 -> not allowed for me = (
    TUPLES=$(
      cat <<'EOF'
normal|48:00:00|64G
EOF
    )
    ;;
esac

# ---------------------------------
# Helpers
# ---------------------------------

contains_in_filter() {
  local part="$1"
  [[ -z "$ONLY_PARTITIONS" ]] && return 0
  IFS=',' read -r -a arr <<<"$ONLY_PARTITIONS"
  for p in "${arr[@]}"; do
    [[ "$p" == "$part" ]] && return 0
  done
  return 1
}

submit_one() {
  local partition="$1" time="$2" mem="$3"

  # Compose sbatch command
  local cmd=( sbatch
    --job-name="$JOB_NAME"
    --partition="$partition"
    --time="$time"
    --gres="gpu:${GPUS}"
    --nodes="$NODES"
    --ntasks="$NTASKS"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$mem"
    --wrap="$(
      cat <<WRAP
set -euo pipefail
$MODULE_LOAD
echo 'Node: ' \$(hostname)
echo 'CUDA visible devices:'
nvidia-smi || true
echo 'Running: ${SCRIPT}'
export WANDB_API_KEY=\${WANDB_API_KEY:-}
export WANDB_ENTITY='${ENTITY}'
export WANDB_PROJECT='${PROJECT}'
export WANDB_NAME='${EXP_NAME}'
cd '${EXPERIMENTS_PATH}'
pixi run python ${PY_ARGS[*]}
WRAP
    )"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

# ---------------------------------
# Main loop
# ---------------------------------
echo "Cluster: $CLUSTER"
echo "Script : $SCRIPT"
echo "Exp    : $EXP_NAME"
echo "DryRun : $DRY_RUN"

while IFS='|' read -r PARTITION TIME MEM; do
  [[ -z "${PARTITION:-}" ]] && continue
  if contains_in_filter "$PARTITION"; then
    echo "Submitting -> partition=${PARTITION} time=${TIME} mem=${MEM}"
    submit_one "$PARTITION" "$TIME" "$MEM"
  else
    echo "Skipping (filtered) -> ${PARTITION}"
  fi
done <<< "$TUPLES"
