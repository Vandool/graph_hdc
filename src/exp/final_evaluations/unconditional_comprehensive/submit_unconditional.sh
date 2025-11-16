#!/bin/bash
#!/usr/bin/env bash
# Cluster-aware submission script for unconditional generation evaluation
# Usage: bash submit_unconditional.sh QM9_SMILES_HRR_1600_F64_G1NG3
# CLUSTER: uc3 | hk | haic | local

set -euo pipefail

# --- DRY-RUN normalization ---
DRY_RUN_RAW="${DRY_RUN:-0}"
DRY_RUN="$(printf '%s' "$DRY_RUN_RAW" | tr -d '\r[:space:]')"
DRY_RUN="${DRY_RUN:-0}"
echo "DryRun  : ${DRY_RUN}"

# -----------------------------
# Parse experiment parameters
# -----------------------------
# QM9_SMILES_HRR_256_F64_G1NG3
# QM9_SMILES_HRR_1600_F64_G1NG3
# ZINC_SMILES_HRR_256_F64_5G1NG4
# ZINC_SMILES_HRR_1024_F64_5G1NG4
# ZINC_SMILES_HRR_2048_F64_5G1NG4
# ZINC_SMILES_HRR_5120_F64_G1G3
DATASET="${1:-ZINC_SMILES_HRR_256_F64_5G1NG4}"

# -----------------------------
# User-configurable parameters
# -----------------------------
CLUSTER="${CLUSTER:-uc3}"

GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-}"   # set per-cluster if empty
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"

N_SAMPLES="${N_SAMPLES:-10000}"
N_SEEDS="${N_SEEDS:-1}"
TIME_LIMIT="${TIME_LIMIT:-}"  # Override time limit if set

MODULE_LOAD_DEFAULT=''
ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"

# Paths
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/final_evaluations/unconditional_comprehensive}"
SCRIPT_NAME="${SCRIPT_NAME:-eval_unconditional_comprehensive.py}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

echo "Script  : ${SCRIPT}"
echo "Dataset : ${DATASET}"
echo "N_Samples: ${N_SAMPLES}"
echo "N_Seeds : ${N_SEEDS}"

EXP_NAME="Eval_cond_${DATASET}_S${N_SAMPLES}"

# -----------------------------
# Build python args (array) + safely quoted string
# -----------------------------
PY_ARGS=(
  "$SCRIPT"
  --dataset "$DATASET"
  --n_samples "$N_SAMPLES"
  --n_seeds "$N_SEEDS"
  --seed 42
)

# Add extra args if provided (e.g., --draw)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  PY_ARGS+=($EXTRA_ARGS)
fi

QUOTED_ARGS="$(printf '%q ' "${PY_ARGS[@]}")"

# -----------------------------
# Partitions + Pixi env per cluster
# -----------------------------
# Set defaults if TIME_LIMIT not provided
if [[ -z "$TIME_LIMIT" ]]; then
    if [[ "$N_SAMPLES" -le 100 ]]; then
        DEFAULT_TIME="04:00:00"
    elif [[ "$N_SAMPLES" -le 1000 ]]; then
        DEFAULT_TIME="12:00:00"
    else
        DEFAULT_TIME="48:00:00"
    fi
else
    DEFAULT_TIME="$TIME_LIMIT"
fi

case "$CLUSTER" in
  local)
    MODULE_LOAD="$MODULE_LOAD_DEFAULT"
    PIXI_ENV="local"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES="debug|${DEFAULT_TIME}|12G"
    ;;
  uc3)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES="gpu_h100|${DEFAULT_TIME}|32G"
    ;;
  hk)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES=$"accelerated|${DEFAULT_TIME}|64G\naccelerated-h100|${DEFAULT_TIME}|64G"
    ;;
  haic|*)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES="normal|${DEFAULT_TIME}|64G"
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
echo 'Job ID  : \$SLURM_JOB_ID'
echo 'Node    : '\$(hostname)
echo 'CUDA visible devices:'; nvidia-smi || true
echo 'Running : ${SCRIPT}'
echo 'Dataset : ${DATASET}'
echo 'N_Samples: ${N_SAMPLES}'
echo 'N_Seeds : ${N_SEEDS}'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
cd '${PROJECT_DIR}'
pixi run --frozen -e '${PIXI_ENV}' python ${QUOTED_ARGS}
echo 'Job completed successfully!'
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
