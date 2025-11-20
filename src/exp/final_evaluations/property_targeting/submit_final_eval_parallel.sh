#!/bin/bash

# Property Targeting Final Evaluation - Parallel Submission Script
# =================================================================
#
# This script submits individual SLURM jobs for each target value,
# enabling parallel execution of final evaluations.
#
# Usage:
#   # Basic usage
#   bash submit_final_eval_parallel.sh <hpo_dir>
#
#   # With custom parameters
#   N_SAMPLES=10000 bash submit_final_eval_parallel.sh hpo_results_bw3/logp_QM9_...
#
#   # With molecule drawings
#   N_SAMPLES=10000 DRAW=1 MAX_DRAW=200 bash submit_final_eval_parallel.sh <hpo_dir>
#
#   # Dry run
#   DRY_RUN=1 bash submit_final_eval_parallel.sh <hpo_dir>
#
#   # Specific cluster
#   CLUSTER=uc3 ONLY_PARTITIONS=gpu_h100 bash submit_final_eval_parallel.sh <hpo_dir>

# -----------------------------
# Parameters
# -----------------------------
HPO_DIR=${1}

if [[ -z "$HPO_DIR" ]]; then
  echo "ERROR: HPO directory required." >&2
  echo "Usage: bash submit_final_eval_parallel.sh <hpo_dir>" >&2
  exit 1
fi

if [[ ! -d "$HPO_DIR" ]]; then
  echo "ERROR: HPO directory not found: $HPO_DIR" >&2
  exit 1
fi

# Parse HPO directory structure to extract targets
TARGET_DIRS=($(find "$HPO_DIR" -maxdepth 1 -type d -name "target_*" | sort))

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No target directories found in $HPO_DIR" >&2
  echo "Expected format: target_<value>/" >&2
  exit 1
fi

# Extract target values from directory names
TARGETS=()
for dir in "${TARGET_DIRS[@]}"; do
  target_name=$(basename "$dir")
  target_value=${target_name#target_}
  TARGETS+=("$target_value")
done

# Configuration from environment variables
N_SAMPLES=${N_SAMPLES:-10000}
DRAW=${DRAW:-0}
MAX_DRAW=${MAX_DRAW:-200}
OUTPUT_DIR=${OUTPUT_DIR:-final_results}
DRY_RUN=${DRY_RUN:-0}

echo "====================================="
echo "Property Targeting Final Evaluation"
echo "====================================="
echo "HPO Directory: $HPO_DIR"
echo "Targets Found: ${TARGETS[*]}"
echo "N Samples per target: $N_SAMPLES"
echo "Draw Molecules: $DRAW"
if [[ "$DRAW" == "1" ]]; then
  echo "Max Draw: $MAX_DRAW"
fi
echo "Output Directory: $OUTPUT_DIR"
echo "Num Jobs to Submit: ${#TARGETS[@]} (one per target)"
echo "====================================="

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/final_evaluations/property_targeting}"
SCRIPT="run_property_targeting_final_eval.py"
SCRIPT_PATH="${EXPERIMENTS_PATH}/${SCRIPT}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Script: ${SCRIPT_PATH}"

# -----------------------------
# Cluster Configuration
# -----------------------------
CLUSTER=${CLUSTER:-local}

case "$CLUSTER" in
  local)
    MODULE_LOAD="${MODULE_LOAD_DEFAULT:-}"
    PIXI_ENV="local"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES=$'debug|48:00:00|16G'
    ;;
  uc3)
    MODULE_LOAD=""  # No CUDA module needed for CPU nodes
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES=$'cpu_il|36:00:00|128G\ncpu|36:00:00|128G'
    ;;
  hk)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'accelerated|72:00:00|64G\naccelerated-h100|72:00:00|64G'
    ;;
  haic|*)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'normal|96:00:00|64G'
    ;;
esac

# Normalize tuples
TUPLES="$(printf '%s' "${TUPLES:-}" | tr -d '\r')"
[[ "$DRY_RUN" == "1" ]] && { echo "---- TUPLES ----"; printf '%s\n' "$TUPLES"; echo "-----------------"; }

# -----------------------------
# Helper Functions
# -----------------------------
contains_in_filter() {
  local part="$1"
  [[ -z "$ONLY_PARTITIONS" ]] && return 0
  IFS=',' read -r -a arr <<<"$ONLY_PARTITIONS"
  for p in "${arr[@]}"; do [[ "$p" == "$part" ]] && return 0; done
  return 1
}

submit_target() {
  local target="$1"
  local partition="$2"
  local time="$3"
  local mem="$4"

  # Build job name
  local hpo_dirname=$(basename "$HPO_DIR")
  local job_name="PropTargetFinal-${target} ${HPO_DIR}"
  local exp_name="PropTarget Final: ${hpo_dirname} target=${target}"

  # Build Python args
  local py_args=(
    "$SCRIPT_PATH"
    --hpo_dir "$HPO_DIR"
    --target "$target"
    --n_samples "$N_SAMPLES"
  )

  if [[ "$DRAW" == "1" ]]; then
    py_args+=(--draw --max_draw "$MAX_DRAW")
  fi

  if [[ -n "$OUTPUT_DIR" ]]; then
    py_args+=(--output_dir "$OUTPUT_DIR")
  fi

  local quoted_args
  quoted_args="$(printf '%q ' "${py_args[@]}")"

  local cmd=( sbatch
    --job-name="$job_name"
    --partition="$partition"
    --time="$time"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --cpu-bind=cores
    --mem="$mem"
    --wrap="$(
      cat <<WRAP
set -euo pipefail
$MODULE_LOAD
echo 'Experiment: ${exp_name}'
echo 'Node:' \$(hostname)
echo 'Running: ${SCRIPT}'
echo 'HPO Dir: ${HPO_DIR}'
echo 'Target: ${target}'
export OMP_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export PYTORCH_NUM_THREADS=$CPUS_PER_TASK
cd '${EXPERIMENTS_PATH}'
pixi run --frozen -e '${PIXI_ENV}' python ${quoted_args}
WRAP
    )"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] Target=%s ' "$target"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  echo "  Submitting target=$target to $partition..."
  "${cmd[@]}"
}

# -----------------------------
# Main Submission
# -----------------------------
echo ""
echo "Cluster: $CLUSTER"
echo "Pixi Env: ${PIXI_ENV}"
echo ""

# Read partition configurations into array
PARTITION_CONFIGS=()
while IFS='|' read -r PARTITION TIME MEM; do
  [[ -z "${PARTITION:-}" ]] && continue
  if contains_in_filter "$PARTITION"; then
    PARTITION_CONFIGS+=("$PARTITION|$TIME|$MEM")
  fi
done <<< "$TUPLES"

if [[ ${#PARTITION_CONFIGS[@]} -eq 0 ]]; then
  echo "ERROR: No valid partition configurations after filtering." >&2
  exit 1
fi

# Use first partition configuration for all jobs
PARTITION_CONFIG="${PARTITION_CONFIGS[0]}"
IFS='|' read -r PARTITION TIME MEM <<< "$PARTITION_CONFIG"

echo "Using partition configuration: partition=$PARTITION time=$TIME mem=$MEM cpus=$CPUS_PER_TASK"
echo ""
echo "Submitting ${#TARGETS[@]} jobs (one per target)..."
echo ""

# Submit one job per target
for target in "${TARGETS[@]}"; do
  submit_target "$target" "$PARTITION" "$TIME" "$MEM"
done

echo ""
echo "Submission complete!"
echo "Submitted ${#TARGETS[@]} jobs for targets: ${TARGETS[*]}"
echo ""
