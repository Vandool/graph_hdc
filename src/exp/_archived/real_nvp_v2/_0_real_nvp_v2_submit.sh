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

JOB_NAME="${JOB_NAME:-RealNVP_v2}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-}"   # set per-cluster if empty
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"

MODULE_LOAD_DEFAULT=''

# Paths
PROJECT_DIR="${PROJECT_DIR:-${GHDC_HOME:-$PWD}}"
EXPERIMENTS_PATH="${EXPERIMENTS_PATH:-${PROJECT_DIR}/src/exp/real_nvp_v2}"
SCRIPT_NAME="${SCRIPT_NAME:-0_real_nvp_v2.py}"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"
echo "Script  : ${SCRIPT}"

# Logging / naming
EXP_NAME="${EXP_NAME:-}"        # if empty we build a deterministic one

ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"

# Convenience resume toggle (adds --continue_from if last.ckpt exists)
RESUME="${RESUME:-0}"

# -----------------------------
# Args (must match get_flow_cli_args)
# -----------------------------
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VSA="${VSA:-HRR}"                 # optional; your parser uses Enum type=VSAModel
DATASET="${DATASET:-}"            # optional (SUPPRESS by default)
DS_TAG="${DS_TAG:-}"            # optional (SUPPRESS by default)
HV_DIM="${HV_DIM:-7744}"          # default 88*88
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

IS_DEV="${IS_DEV:-False}"


NUM_FLOWS="${NUM_FLOWS:-8}"
NUM_HIDDEN_CHANNELS="${NUM_HIDDEN_CHANNELS:-512}"
USE_ACT_NORM="${USE_ACT_NORM:-1}"       # 1 to add flag, 0 to skip

SMAX_INITIAL="${SMAX_INITIAL:-1.0}"
SMAX_FINAL="${SMAX_FINAL:-6.0}"
SMAX_WARMUP_EPOCHS="${SMAX_WARMUP_EPOCHS:-15}"

CONTINUE_FROM="${CONTINUE_FROM:-}"      # explicit path beats RESUME discovery
RESUME_RETRAIN_LAST_EPOCH="${RESUME_RETRAIN_LAST_EPOCH:-0}"  # adds flag if 1

# -----------------------------
# Deterministic EXP_NAME (if empty)
# -----------------------------
to_token() { echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'; }
if [[ -z "$EXP_NAME" ]]; then
  LR_TOK=$(to_token "$LR")
  WD_TOK=$(to_token "$WEIGHT_DECAY")
  AN_TOK=$([ "${USE_ACT_NORM}" = "1" ] && echo "an" || echo "noan")
  VSA_TAG=$(echo "${VSA:0:1}" | tr '[:upper:]' '[:lower:]')
  EXP_NAME="nvp_${DS_TAG}_${VSA_TAG}${HV_DIM}_f${NUM_FLOWS}_hid${NUM_HIDDEN_CHANNELS}_s${SEED}_lr${LR_TOK}_wd${WD_TOK}_${AN_TOK}"
fi

# -----------------------------
# Build python args (array) + safely quoted string
# -----------------------------
PY_ARGS=(
  "$SCRIPT"
  --exp_dir_name "$EXP_NAME"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --hv_dim "$HV_DIM"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --num_flows "$NUM_FLOWS"
  --num_hidden_channels "$NUM_HIDDEN_CHANNELS"
  --smax_initial "$SMAX_INITIAL"
  --smax_final "$SMAX_FINAL"
  --smax_warmup_epochs "$SMAX_WARMUP_EPOCHS"
  --is_dev "$IS_DEV"
)

# Optional args per your CLI (SUPPRESS by default)
[[ -n "$VSA" ]] && PY_ARGS+=( --vsa "$VSA" )
[[ -n "$DATASET" ]] && PY_ARGS+=( --dataset "$DATASET" )

# Flag-style args (only include when true)
[[ "$USE_ACT_NORM" = "1" ]] && PY_ARGS+=( --use_act_norm )
[[ "$RESUME_RETRAIN_LAST_EPOCH" = "1" ]] && PY_ARGS+=( --resume_retrain_last_epoch )

# Continue-from (explicit path or discovered last.ckpt when RESUME=1)
if [[ -n "$CONTINUE_FROM" ]]; then
  PY_ARGS+=( --continue_from "$CONTINUE_FROM" )
elif [[ "$RESUME" = "1" ]]; then
  CKPT_PATH="${EXPERIMENTS_PATH}/results/0_real_nvp_v2/${EXP_NAME}/models/last.ckpt"
  [[ -f "$CKPT_PATH" ]] && PY_ARGS+=( --continue_from "$CKPT_PATH" )
fi

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
    TUPLES=$'gpu_h100|24:00:00|64G\ngpu_a100_il|24:00:00|64G\ngpu_h100_il|24:00:00|64G'
    ;;
  hk)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'accelerated|24:00:00|64G\naccelerated-h100|24:00:00|64G'
    ;;
  haic|*)
    MODULE_LOAD="module load devel/cuda"
    PIXI_ENV="cluster"
    [[ -z "${CPUS_PER_TASK:-}" ]] && CPUS_PER_TASK=16
    TUPLES=$'normal|24:00:00|64G'
    ;;
esac

# Filter (optional)
ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"

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
export WANDB_API_KEY=\${WANDB_API_KEY:-}
export WANDB_ENTITY='arvand-kaveh-karlsruhe-institute-of-technology'
export WANDB_PROJECT='real_nvp_v2'
export WANDB_NAME='${EXP_NAME}'
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
