#!/bin/bash

 # Parallel Property Targeting Submission Script (MG-DIFF Protocol)
# ==================================================================
#
# This script submits individual SLURM jobs for each property target value,
# enabling parallel execution instead of sequential processing.
#
# Usage:
#   # HPO mode - submit all targets for a property in parallel
#   bash submit_property_targeting_parallel.sh hpo logp QM9_SMILES_HRR_256_F64_G1NG3
#   bash submit_property_targeting_parallel.sh hpo qed ZINC_SMILES_HRR_1024_F64_5G1NG4
#
#   # Final evaluation mode - process all targets from HPO directory
#   bash submit_property_targeting_parallel.sh final path/to/hpo_results/logp_QM9_SMILES_HRR_1600_F64_G1NG3_20251113_120000
#
#   # With custom parameters
#   N_TRIALS=100 N_SAMPLES=1000 bash submit_property_targeting_parallel.sh hpo logp QM9_SMILES_HRR_1600_F64_G1NG3
#
#   # With specific cluster configuration
#   CLUSTER=uc3 ONLY_PARTITIONS=gpu_h100 bash submit_property_targeting_parallel.sh hpo qed ZINC_SMILES_HRR_2048_F64_5G1NG4
#
#   # Dry run to preview job submissions
#   DRY_RUN=1 bash submit_property_targeting_parallel.sh hpo logp QM9_SMILES_HRR_256_F64_G1NG3

# -----------------------------
# Mode and Parameters
# -----------------------------
MODE=${1:-hpo}  # "hpo" or "final"

if [[ "$MODE" != "hpo" && "$MODE" != "final" ]]; then
  echo "ERROR: Invalid mode '$MODE'. Must be 'hpo' or 'final'." >&2
  echo "Usage: bash submit_property_targeting_parallel.sh <hpo|final> [args...]" >&2
  exit 1
fi

# -----------------------------
# HPO Mode Parameters
# -----------------------------
# ZINC_SMILES_HRR_256_F64_5G1NG4
# QM9_SMILES_HRR_256_F64_G1NG3
if [[ "$MODE" == "hpo" ]]; then
  PROPERTY=${2:-logp}
  DATASET=${3:-ZINC_SMILES_HRR_256_F64_5G1NG4}
  MODEL_IDX=${4:-0}
  N_TRIALS=${N_TRIALS:-1}
  N_SAMPLES=${N_SAMPLES:-20}
  OUTPUT_DIR=${OUTPUT_DIR:-hpo_results}

  # Determine dataset family (zinc or qm9) from dataset name
  if [[ "$DATASET" == *"ZINC"* || "$DATASET" == *"Zinc"* || "$DATASET" == *"zinc"* ]]; then
    DATASET_FAMILY="zinc"
  elif [[ "$DATASET" == *"QM9"* || "$DATASET" == *"qm9"* ]]; then
    DATASET_FAMILY="qm9"
  else
    echo "ERROR: Cannot determine dataset family from '$DATASET'." >&2
    echo "Dataset name must contain 'ZINC' or 'QM9'." >&2
    exit 1
  fi

  # MG-DIFF Target Values
  # These match the MGDIFF_TARGETS dictionary in run_property_targeting_hpo.py:
  # MGDIFF_TARGETS = {
  #     "zinc": {"logp": [2.0, 4.0, 6.0], "qed": [0.6, 0.75, 0.9], "sa_score": [2.0, 3.0, 4.0], "tpsa": [30.0, 60.0, 90.0]},
  #     "qm9": {"logp": [-1, 0.5, 2.0], "qed": [0.3, 0.45, 0.6], "sa_score": [3.0, 4.5, 6.0], "tpsa": [30.0, 60.0, 90.0]},
  # }
  case "${DATASET_FAMILY}-${PROPERTY}" in
    zinc-logp)
      TARGETS=(2.0 4.0 6.0)
      ;;
    zinc-qed)
      TARGETS=(0.6 0.75 0.9)
      ;;
    zinc-sa_score)
      TARGETS=(2.0 3.0 4.0)
      ;;
    zinc-tpsa)
      TARGETS=(30.0 60.0 90.0)
      ;;
    qm9-logp)
      TARGETS=(-1 0.5 2.0)
      ;;
    qm9-qed)
      TARGETS=(0.3 0.45 0.6)
      ;;
    qm9-sa_score)
      TARGETS=(3.0 4.5 6.0)
      ;;
    qm9-tpsa)
      TARGETS=(30.0 60.0 90.0)
      ;;
    *)
      echo "ERROR: Unknown property '$PROPERTY' for dataset family '$DATASET_FAMILY'." >&2
      echo "Supported properties: logp, qed, sa_score, tpsa" >&2
      exit 1
      ;;
  esac

  echo "====================================="
  echo "Property Targeting HPO Submission (PARALLEL)"
  echo "====================================="
  echo "Property: $PROPERTY"
  echo "Dataset: $DATASET (family: $DATASET_FAMILY)"
  echo "Model Index: $MODEL_IDX"
  echo "Targets: ${TARGETS[*]}"
  echo "N Trials: $N_TRIALS"
  echo "N Samples: $N_SAMPLES"
  echo "Output Directory: $OUTPUT_DIR"
  echo "Num Jobs to Submit: ${#TARGETS[@]}"
  echo "====================================="

# -----------------------------
# Final Evaluation Mode Parameters
# -----------------------------
elif [[ "$MODE" == "final" ]]; then
  HPO_DIR=${2}

  if [[ -z "$HPO_DIR" ]]; then
    echo "ERROR: HPO directory required for final mode." >&2
    echo "Usage: bash submit_property_targeting_parallel.sh final <hpo_dir>" >&2
    exit 1
  fi

  if [[ ! -d "$HPO_DIR" ]]; then
    echo "ERROR: HPO directory not found: $HPO_DIR" >&2
    exit 1
  fi

  # Parse HPO directory structure to extract targets
  # Expected format: {property}_{dataset}_{timestamp}/target_{value}/
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

  N_SAMPLES=${N_SAMPLES:-100}
  DRAW=${DRAW:-0}
  MAX_DRAW=${MAX_DRAW:-100}
  OUTPUT_DIR=${OUTPUT_DIR:-hpo_results}

  echo "====================================="
  echo "Property Targeting Final Evaluation (PARALLEL)"
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
fi

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/final_evaluations/property_targeting}"

if [[ "$MODE" == "hpo" ]]; then
  SCRIPT="run_property_targeting_hpo.py"
else
  SCRIPT="run_property_targeting_final_eval.py"
fi

SCRIPT_PATH="${EXPERIMENTS_PATH}/${SCRIPT}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Script: ${SCRIPT_PATH}"

# -----------------------------
# Cluster Configuration
# -----------------------------
CLUSTER=${CLUSTER:-uc3}

case "$CLUSTER" in
  local)
    MODULE_LOAD="${MODULE_LOAD_DEFAULT:-}"
    PIXI_ENV="local"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=4
    TUPLES=$'debug|48:00:00|16G'
    ;;
  uc3)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    # Increase time for final evaluation (10k samples takes longer)
    if [[ "$MODE" == "final" ]]; then
      TUPLES=$'gpu_h100|96:00:00|64G\ngpu_a100_il|96:00:00|64G\ngpu_h100_il|96:00:00|64G'
    else
      TUPLES=$'gpu_h100|72:00:00|64G\ngpu_a100_il|72:00:00|64G\ngpu_h100_il|72:00:00|64G'
    fi
    ;;
  hk)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    if [[ "$MODE" == "final" ]]; then
      TUPLES=$'accelerated|72:00:00|64G\naccelerated-h100|72:00:00|64G'
    else
      TUPLES=$'accelerated|48:00:00|64G\naccelerated-h100|48:00:00|64G'
    fi
    ;;
  haic|*)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    if [[ "$MODE" == "final" ]]; then
      TUPLES=$'normal|96:00:00|64G'
    else
      TUPLES=$'normal|72:00:00|64G'
    fi
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

submit_one_target() {
  local target="$1"
  local partition="$2"
  local time="$3"
  local mem="$4"

  # Build job name with target value (HPO mode only)
  local job_name="PropTarget-${PROPERTY}-${target}"
  local exp_name="PropTarget HPO ${PROPERTY}=${target} ${DATASET}"

  # Build Python args for this specific target (HPO mode only)
  local py_args=(
    "$SCRIPT_PATH"
    --dataset "$DATASET"
    --property "$PROPERTY"
    --model_idx "$MODEL_IDX"
    --n_samples "$N_SAMPLES"
    --n_trials "$N_TRIALS"
    --targets "$target"
    --output_dir "$OUTPUT_DIR"
  )

  local quoted_args
  quoted_args="$(printf '%q ' "${py_args[@]}")"

  local cmd=( sbatch
    --job-name="$job_name"
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
echo 'Experiment: ${exp_name}'
echo 'Node:' \$(hostname)
echo 'CUDA visible devices:'; nvidia-smi || true
echo 'Running: ${SCRIPT}'
echo 'Target: ${target}'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
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

submit_final_eval() {
  local target="$1"
  local partition="$2"
  local time="$3"
  local mem="$4"

  # Extract property and dataset from HPO directory name
  local hpo_dirname=$(basename "$HPO_DIR")
  local job_name="PropTargetFinal-${target}"
  local exp_name="PropTarget Final Eval: ${hpo_dirname} target=${target}"

  # Build Python args for final evaluation
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
    --gres="gpu:1"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$mem"
    --wrap="$(
      cat <<WRAP
set -euo pipefail
$MODULE_LOAD
echo 'Experiment: ${exp_name}'
echo 'Node:' \$(hostname)
echo 'CUDA visible devices:'; nvidia-smi || true
echo 'Running: ${SCRIPT}'
echo 'HPO Dir: ${HPO_DIR}'
echo 'Target: ${target}'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
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
echo "Mode: $MODE"
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
# (You can modify this to round-robin or use different strategies)
PARTITION_CONFIG="${PARTITION_CONFIGS[0]}"
IFS='|' read -r PARTITION TIME MEM <<< "$PARTITION_CONFIG"

echo "Using partition configuration: partition=$PARTITION time=$TIME mem=$MEM cpus=$CPUS_PER_TASK"
echo ""

echo "Submitting ${#TARGETS[@]} jobs (one per target)..."
echo ""

# Submit one job per target
for target in "${TARGETS[@]}"; do
  if [[ "$MODE" == "hpo" ]]; then
    submit_one_target "$target" "$PARTITION" "$TIME" "$MEM"
  else
    submit_final_eval "$target" "$PARTITION" "$TIME" "$MEM"
  fi
done

echo ""
echo "Submission complete!"
echo "Submitted ${#TARGETS[@]} jobs for targets: ${TARGETS[*]}"
echo ""
