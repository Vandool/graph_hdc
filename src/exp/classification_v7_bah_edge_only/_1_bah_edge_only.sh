#!/usr/bin/env bash
# universal_cluster_submit.sh
# Submit the same experiment to multiple partitions depending on $CLUSTER.
# CLUSTER: uc3 | hk | haic | local

set -euo pipefail

# --- DRY-RUN normalization & early debug ---
DRY_RUN_RAW="${DRY_RUN:-0}"
DRY_RUN="$(printf '%s' "$DRY_RUN_RAW" | tr -d '\r[:space:]')"
DRY_RUN="${DRY_RUN:-0}"
echo "DryRun  : ${DRY_RUN}"

# -----------------------------
# User-configurable parameters
# -----------------------------
CLUSTER="${CLUSTER:-local}"

JOB_NAME="${JOB_NAME:-BAH_Lightning}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-}"   # set by cluster block if empty
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"

MODULE_LOAD_DEFAULT=''

PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/classification_v7_bah_edge_only/bah_edge_only.py}"
SCRIPT_NAME="${SCRIPT_NAME:-bah_edge_only.py}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"
echo "Script  : ${SCRIPT}"

ENTITY="${ENTITY:-akaveh}"
PROJECT="${PROJECT:-graph_hdc}"
EXP_NAME="${EXP_NAME:-$JOB_NAME}"

ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"

shopt -s nocasematch
IS_DEV="${IS_DEV:-False}"
if [[ "$IS_DEV" =~ ^(1|true|yes|on)$ ]]; then
  EXP_NAME="DEBUG_${EXP_NAME}"
  JOB_NAME="DEBUG_${JOB_NAME}"
fi
shopt -u nocasematch

# -----------------------------
# Args
# -----------------------------
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_NAME="${MODEL_NAME:-biaffine_head}"
IS_DEV="${IS_DEV:-False}"

# Biaffine head
PROJ_DIM="${PROJ_DIM:-1024}"
N_HEADS="${N_HEADS:-8}"
PROJ_hidden="${PROJ_hidden:-}"          # empty -> None
DROPOUT="${DROPOUT:-0.0}"
SHARE_PROJ="${SHARE_PROJ:-False}"
NORM="${NORM:-True}"
USE_LAYERNORM="${USE_LAYERNORM:-True}"
USE_TEMPERATURE="${USE_TEMPERATURE:-True}"
POS_WEIGHT="${POS_WEIGHT:-}"             # empty -> None

# HDC / encoder
HV_DIM="${HV_DIM:-1600}"
VSA="${VSA:-HRR}"
DATASET="${DATASET:-QM9_SMILES_HRR_1600}"

# Optim
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# Loader
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-6}"
PIN_MEMORY="${PIN_MEMORY:-True}"
MICRO_BS="${MICRO_BS:-64}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-True}"

# Checkpointing
CONTINUE_FROM="${CONTINUE_FROM:-}"
RESUME_RETRAIN_LAST_EPOCH="${RESUME_RETRAIN_LAST_EPOCH:-False}"

# Stratification
STRATIFY="${STRATIFY:-True}"
RESAMPLE_TRAINING_DATA_ON_BATCH="${RESAMPLE_TRAINING_DATA_ON_BATCH:-False}"

# -----------------------------
# Build python args (array) + safe quoted string
# -----------------------------
PY_ARGS=(
  "$SCRIPT"
  --project_dir "$PROJECT_DIR"
  --exp_dir_name "$EXP_NAME"

  --seed "$SEED"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --model_name "$MODEL_NAME"
  --is_dev "$IS_DEV"

  --proj_dim "$PROJ_DIM"
  --n_heads "$N_HEADS"
  --dropout "$DROPOUT"
  --share_proj "$SHARE_PROJ"
  --norm "$NORM"
  --use_layernorm "$USE_LAYERNORM"
  --use_temperature "$USE_TEMPERATURE"

  --hv_dim "$HV_DIM"
  --vsa "$VSA"
  --dataset "$DATASET"

  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"

  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --pin_memory "$PIN_MEMORY"
  --persistent_workers "$PERSISTENT_WORKERS"
  --micro_bs "$MICRO_BS"

  --resume_retrain_last_epoch "$RESUME_RETRAIN_LAST_EPOCH"
)

# Optional / nullable args
[[ -n "$PROJ_hidden" ]] && PY_ARGS+=( --proj_hidden "$PROJ_hidden" )
[[ -n "$CONTINUE_FROM" ]] && PY_ARGS+=( --continue_from "$CONTINUE_FROM" )

# Quote each arg so --wrap re-splits correctly on the remote shell
QUOTED_ARGS="$(printf '%q ' "${PY_ARGS[@]}")"
# -----------------------------
# Partitions + Pixi env per cluster
# -----------------------------
case "$CLUSTER" in
  local)
    MODULE_LOAD="$MODULE_LOAD_DEFAULT"
    PIXI_ENV="local"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES=$'debug|00:10:00|8G'
    ;;
  uc3)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'gpu_h100|36:00:00|64G\ngpu_a100_il|36:00:00|64G\ngpu_h100_il|36:00:00|64G'
    ;;
  hk)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'accelerated|36:00:00|64G\naccelerated-h100|36:00:00|64G'
    ;;
  haic|*)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'normal|48:00:00|64G'
    ;;
esac

# Normalize TUPLES newlines for portability
TUPLES="$(printf '%s' "${TUPLES:-}" | tr -d '\r')"
[[ "$DRY_RUN" == "1" ]] && { echo "---- TUPLES ----"; printf '%s\n' "$TUPLES"; echo "-----------------"; }

# -----------------------------
# Helpers
# -----------------------------
contains_in_filter() {
  local part="$1"
  [[ -z "$ONLY_PARTITIONS" ]] && return 0
  IFS=',' read -r -a arr <<<"$ONLY_PARTITIONS"
  for p in "${arr[@]}"; do [[ "$p" == "$part" ]] && return 0; done
  return 1
}

submit_one() {
  local partition="$1" time="$2" mem="$3"

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
echo 'Node:' \$(hostname)
echo 'CUDA visible devices:'; nvidia-smi || true
echo 'Running: ${SCRIPT}'
export WANDB_API_KEY=\${WANDB_API_KEY:-}
export WANDB_ENTITY='${ENTITY}'
export WANDB_PROJECT='${PROJECT}'
export WANDB_NAME='${EXP_NAME}'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
cd '${EXPERIMENTS_PATH}'
# Use Pixi lockfile (frozen) and pick env based on cluster/local
pixi run --frozen -e '${PIXI_ENV}' python ${QUOTED_ARGS}
WRAP
    )"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '; printf '%q ' "${cmd[@]}"; printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

# -----------------------------
# Main
# -----------------------------
echo "Cluster : $CLUSTER"
echo "PixiEnv : ${PIXI_ENV}"
echo "Exp     : $EXP_NAME"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: Script not found: $SCRIPT" >&2
  exit 1
fi

while IFS='|' read -r PARTITION TIME MEM; do
  [[ -z "${PARTITION:-}" ]] && continue
  echo "Tuple    -> partition='${PARTITION}' time='${TIME}' mem='${MEM}'"
  if contains_in_filter "$PARTITION"; then
    echo "Submitting -> partition=${PARTITION} time=${TIME} mem=${MEM} cpus=${CPUS_PER_TASK}"
    submit_one "$PARTITION" "$TIME" "$MEM"
  else
    echo "Skipping (filtered) -> ${PARTITION}"
  fi
done <<< "$TUPLES"
