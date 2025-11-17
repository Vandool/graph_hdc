#!/bin/bash

# Property Targeting Submission Script (MG-DIFF Protocol)
# ========================================================
#
# This unified script supports both HPO and final evaluation modes.
#
# Usage:
#   # HPO mode
#   bash submit_property_targeting.sh hpo logp QM9_SMILES_HRR_256_F64_G1NG3
#
#   # Final evaluation mode
#   bash submit_property_targeting.sh final path/to/hpo_results/logp_QM9_SMILES_HRR_1600_F64_G1NG3_20251113_120000
#
#   # With custom parameters
#   N_TRIALS=100 N_SAMPLES=1000 bash submit_property_targeting.sh hpo logp QM9_SMILES_HRR_1600_F64_G1NG3
#
#   # Final with drawings
#   DRAW=1 MAX_DRAW=200 bash submit_property_targeting.sh final <hpo_dir>

# -----------------------------
# Mode and Parameters
# -----------------------------
MODE=${1:-hpo}  # "hpo" or "final"

if [[ "$MODE" != "hpo" && "$MODE" != "final" ]]; then
  echo "ERROR: Invalid mode '$MODE'. Must be 'hpo' or 'final'." >&2
  echo "Usage: bash submit_property_targeting.sh <hpo|final> [args...]" >&2
  exit 1
fi

# -----------------------------
# HPO Mode Parameters
# -----------------------------
# ZINC_SMILES_HRR_256_F64_5G1NG4
# QM9_SMILES_HRR_256_F64_G1NG3
if [[ "$MODE" == "hpo" ]]; then
  PROPERTY=${2:-logp}
  DATASET=${3:-QM9_SMILES_HRR_256_F64_G1NG3}
  MODEL_IDX=${4:-0}
  N_TRIALS=${N_TRIALS:-50}
  N_SAMPLES=${N_SAMPLES:-100}


## ===== MG-DIFF Target Values =====
#MGDIFF_TARGETS = {
#    "zinc": {"logp": [2.0, 4.0, 6.0], "qed": [0.6, 0.75, 0.9], "sa_score": [2.0, 3.0, 4.0], "tpsa": [30.0, 60.0, 90.0]},
#    "qm9": {"logp": [-1, 0.5, 2.0], "qed": [0.3, 0.45, 0.6], "sa_score": [3.0, 4.5, 6.0], "tpsa": [30.0, 60.0, 90.0]},
#}


  OUTPUT_DIR=${OUTPUT_DIR:-hpo_results}
  SCRIPT_NAME="run_property_targeting_hpo.py"
  EXP_NAME="PropTarget HPO ${PROPERTY} ${DATASET}"

  echo "====================================="
  echo "Property Targeting HPO Submission"
  echo "====================================="
  echo "Property: $PROPERTY"
  echo "Dataset: $DATASET"
  echo "Model Index: $MODEL_IDX"
  echo "Targets: $TARGETS"
  echo "N Trials: $N_TRIALS"
  echo "N Samples: $N_SAMPLES"
  echo "Output Directory: $OUTPUT_DIR"
  echo "====================================="

# -----------------------------
# Final Evaluation Mode Parameters
# -----------------------------
elif [[ "$MODE" == "final" ]]; then
  HPO_DIR=${2}

  if [[ -z "$HPO_DIR" ]]; then
    echo "ERROR: HPO directory required for final mode." >&2
    echo "Usage: bash submit_property_targeting.sh final <hpo_dir>" >&2
    exit 1
  fi

  if [[ ! -d "$HPO_DIR" ]]; then
    echo "ERROR: HPO directory not found: $HPO_DIR" >&2
    exit 1
  fi

  N_SAMPLES=${N_SAMPLES:-100}
  DRAW=${DRAW:-0}
  MAX_DRAW=${MAX_DRAW:-100}
#  OUTPUT_DIR=${OUTPUT_DIR:-final_results}
  OUTPUT_DIR=${OUTPUT_DIR:-hpo_results}
  SCRIPT_NAME="run_property_targeting_final_eval.py"
  EXP_NAME="PropTarget Final $(basename "$HPO_DIR")"

  echo "====================================="
  echo "Property Targeting Final Evaluation"
  echo "====================================="
  echo "HPO Directory: $HPO_DIR"
  echo "N Samples: $N_SAMPLES"
  echo "Draw Molecules: $DRAW"
  if [[ "$DRAW" == "1" ]]; then
    echo "Max Draw: $MAX_DRAW"
  fi
  echo "Output Directory: $OUTPUT_DIR"
  echo "====================================="
fi

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/final_evaluations/property_targeting}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: Script not found: $SCRIPT" >&2
  exit 1
fi

echo "Script: ${SCRIPT}"

# -----------------------------
# Build Python Args
# -----------------------------
if [[ "$MODE" == "hpo" ]]; then
  PY_ARGS=(
    "$SCRIPT"
    --dataset "$DATASET"
    --property "$PROPERTY"
    --model_idx "$MODEL_IDX"
    --n_samples "$N_SAMPLES"
    --n_trials "$N_TRIALS"
#    --targets $TARGETS
    --output_dir "$OUTPUT_DIR"
  )
elif [[ "$MODE" == "final" ]]; then
  PY_ARGS=(
    "$SCRIPT"
    --hpo_dir "$HPO_DIR"
    --n_samples "$N_SAMPLES"
  )

  if [[ "$DRAW" == "1" ]]; then
    PY_ARGS+=(--draw --max_draw "$MAX_DRAW")
  fi

  if [[ -n "$OUTPUT_DIR" ]]; then
    PY_ARGS+=(--output_dir "$OUTPUT_DIR")
  fi
fi

QUOTED_ARGS="$(printf '%q ' "${PY_ARGS[@]}")"

# -----------------------------
# Cluster Configuration
# -----------------------------
case "$CLUSTER" in
  local)
    MODULE_LOAD="$MODULE_LOAD_DEFAULT"
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
# Main Submission
# -----------------------------
echo ""
echo "Cluster: $CLUSTER"
echo "Pixi Env: ${PIXI_ENV}"
echo "Experiment: $EXP_NAME"
echo ""

while IFS='|' read -r PARTITION TIME MEM; do
  [[ -z "${PARTITION:-}" ]] && continue
  echo "Tuple -> partition='${PARTITION}' time='${TIME}' mem='${MEM}'"
  if contains_in_filter "$PARTITION"; then
    echo "Submitting -> partition=${PARTITION} time=${TIME} mem=${MEM} cpus=${CPUS_PER_TASK}"
    submit_one "$PARTITION" "$TIME" "$MEM"
  else
    echo "Skipping (filtered) -> ${PARTITION}"
  fi
done <<< "$TUPLES"

echo ""
echo "Submission complete!"
echo ""
