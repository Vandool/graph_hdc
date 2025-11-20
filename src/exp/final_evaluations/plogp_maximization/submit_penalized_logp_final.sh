#!/bin/bash

# Penalized LogP Maximization Final Evaluation Submission Script
# Usage: bash submit_penalized_logp_final.sh <dataset> <model_idx> <hpo_dir>

# Default values
DATASET=${1:-ZINC_SMILES_HRR_256_F64_5G1NG4}
MODEL_IDX=${2:-0}
HPO_DIR=${3:-hpo_results}
N_SAMPLES=${N_SAMPLES:-10000}
MAX_HEAVY_ATOMS=${MAX_HEAVY_ATOMS:-38}
OUTPUT_DIR=${OUTPUT_DIR:-final_results}

# Run Final Evaluation
echo "Starting Penalized LogP Maximization Final Evaluation"
echo "Dataset: $DATASET"
echo "Model Index: $MODEL_IDX"
echo "HPO Directory: $HPO_DIR"
echo "N Samples: $N_SAMPLES"
echo "Max Heavy Atoms: $MAX_HEAVY_ATOMS"
echo "Output Directory: $OUTPUT_DIR"
echo "================================"

# Paths
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/final_evaluations/plogp_maximization}"
SCRIPT_NAME="${SCRIPT_NAME:-eval_penalized_logp_final.py}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"
echo "Script  : ${SCRIPT}"

EXP_NAME="pLogP Max Final ${DATASET}"

# -----------------------------
# Build python args (array) + safely quoted string
# -----------------------------
PY_ARGS=(
  "$SCRIPT"
  --hpo_dir "$HPO_DIR"
  --dataset "$DATASET"
  --model_idx "$MODEL_IDX"
  --n_samples "$N_SAMPLES"
  --max_heavy_atoms "$MAX_HEAVY_ATOMS"
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
    TUPLES=$'debug|72:00:00|16G'
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
    TUPLES=$'normal|72:00:00|64G'
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
    --gres="gpu:1"
    --nodes=1
    --ntasks=1
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
