#!/bin/bash
#!/usr/bin/env bash
# universal_cluster_submit.sh — Real NVP v2
# CLUSTER: uc3 | hk | haic | local

set -euo pipefail

# --- DRY-RUN normalization ---
DRY_RUN_RAW="${DRY_RUN:-0}"
DRY_RUN="$(printf '%s' "$DRY_RUN_RAW" | tr -d '\r[:space:]')"
DRY_RUN="${DRY_RUN:-0}"
echo "DryRun  : ${DRY_RUN}"

# -----------------------------
# User-configurable parameters
# -----------------------------
CLUSTER="${CLUSTER:-local}"

GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-}"   # set per-cluster if empty
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"

# Options:
# QM9_SMILES_HRR_1600_F64_G1G3
# QM9_SMILES_HRR_1600_F64_G1NG3
# ZINC_SMILES_HRR_6144_F64_G1G3
DATASET="${DATASET:-QM9_SMILES_HRR_1600_F64_G1G3}"
N_TRIALS="${N_TRIALS:-20}"
IS_DEV="${IS_DEV:-False}"

MODULE_LOAD_DEFAULT=''
ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"


# Paths
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/real_nvp_hpo/hpo}"
SCRIPT_NAME="${SCRIPT_NAME:-optuna_hpo_debug.py}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"
echo "Script  : ${SCRIPT}"

EXP_NAME="REAL NVP HPO ${DATASET}"

# -----------------------------
# Build python args (array) + safely quoted string
# -----------------------------
PY_ARGS=(
  "$SCRIPT"
  --dataset "$DATASET"
  --n_trials "$N_TRIALS"
)

QUOTED_ARGS="$(printf '%q ' "${PY_ARGS[@]}")"

# -----------------------------
# Partitions + Pixi env per cluster
# -----------------------------
case "$CLUSTER" in
  local)
    MODULE_LOAD="$MODULE_LOAD_DEFAULT"
    PIXI_ENV="local"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES=$'debug|01:00:00|32G'
    ;;
  uc3)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'gpu_h100|72:00:00|64G\ngpu_a100_il|72:00:00|64G\ngpu_h100_il|72:00:00|64G'
    ;;
  hk)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'accelerated|48:00:00|64G\naccelerated-h100|48:00:00|64G'
    ;;
  haic|*)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'normal|72:00:00|196G'
    ;;
esac

# Normalize tuples for portability
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

# W&B logging to project=/home/iti/zi9629 entity=arvand-kaveh-karlsruhe-institute-of-technology name=nvp_small_actnorm_qm9
submit_one() {
  local partition="$1" time="$2" mem="$3"

  local cmd=( sbatch
    --job-name="$EXP_NAME"
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
cd '${EXPERIMENTS_PATH}'
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
