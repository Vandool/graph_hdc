#!/bin/bash
#!/usr/bin/env bash
# Cluster-aware submission script for a single retrieval experiment
# Usage: bash submit_single_job.sh HRR 1024 3 qm9 1
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
VSA="${1:-HRR}"
HV_DIM="${2:-1024}"
DEPTH="${3:-3}"
DATASET="${4:-qm9}"
ITER_BUDGET="${5:-1}"

# -----------------------------
# User-configurable parameters
# -----------------------------
CLUSTER="${CLUSTER:-uc3}"

GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-}"   # set per-cluster if empty
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"

N_SAMPLES="${N_SAMPLES:-1000}"

MODULE_LOAD_DEFAULT=''
ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"

# Paths
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/evals/retrieval_experiments}"
SCRIPT_NAME="${SCRIPT_NAME:-run_retrieval_experiment.py}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENTS_PATH}/results}"

echo "Script  : ${SCRIPT}"
echo "VSA     : ${VSA}"
echo "HV_DIM  : ${HV_DIM}"
echo "DEPTH   : ${DEPTH}"
echo "DATASET : ${DATASET}"
echo "ITER_BDG: ${ITER_BUDGET}"

EXP_NAME="Retrieval_${VSA}_${DATASET}_d${HV_DIM}_D${DEPTH}_i${ITER_BUDGET}"

# -----------------------------
# Build python args (array) + safely quoted string
# -----------------------------
PY_ARGS=(
  "$SCRIPT"
  --vsa "$VSA"
  --hv_dim "$HV_DIM"
  --depth "$DEPTH"
  --dataset "$DATASET"
  --iter_budget "$ITER_BUDGET"
  --n_samples "$N_SAMPLES"
  --output_dir "$OUTPUT_DIR"
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
    TUPLES=$'debug|04:00:00|32G'
    ;;
  uc3)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'gpu_h100|06:00:00|64G\ngpu_a100_il|06:00:00|64G\ngpu_h100_il|06:00:00|64G'
    ;;
  hk)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'accelerated|06:00:00|64G\naccelerated-h100|06:00:00|64G'
    ;;
  haic|*)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'normal|06:00:00|64G'
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
echo 'VSA     : ${VSA}'
echo 'HV_DIM  : ${HV_DIM}'
echo 'DEPTH   : ${DEPTH}'
echo 'DATASET : ${DATASET}'
echo 'ITER_BDG: ${ITER_BUDGET}'
echo 'N_SAMP  : ${N_SAMPLES}'
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
