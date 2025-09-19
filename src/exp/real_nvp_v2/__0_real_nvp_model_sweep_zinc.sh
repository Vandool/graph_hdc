#!/usr/bin/env bash
set -euo pipefail

# Which submit wrapper to call
SUBMIT="${SUBMIT:-./_0_real_nvp_v2_submit.sh}"
[[ ! -x "$SUBMIT" ]] && {
  echo "ERROR: $SUBMIT not found or not executable" >&2
  echo "Hint: ls -1 _0_real_nvp_v2*.sh" >&2
  exit 1
}

# --- dataset & epochs (override via env when calling this script) ---
DATASET="${DATASET:-ZINC_SMILES_HRR_7744}"   # or ZINC_SMILES_HRR_7744
EPOCHS="${EPOCHS:-600}"
DEVICE="${DEVICE:-cuda}"
IS_DEV="${IS_DEV:-False}"

# Infer short dataset tag for naming
DS_LOWER="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')"
if [[ "$DS_LOWER" == *qm9* ]]; then
  DS_TAG="qm9"
elif [[ "$DS_LOWER" == *zinc* ]]; then
  DS_TAG="zinc"
else
  DS_TAG="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"
fi

# Pick HV_DIM if not provided
if [[ -z "${HV_DIM:-}" ]]; then
  if [[ "$DS_TAG" == "zinc" ]]; then
    HV_DIM=$((88*88))   # 7744 typical for ZINC HRR
  else
    HV_DIM=$((40*40))   # 1600 typical for QM9 HRR
  fi
fi

# ---------- Already trained---
#baseline|8|512|1e-3|1|0.0
#baseline_lr5e4|8|512|5e-4|1|0.0
#baseline_wd|8|512|1e-3|1|1e-4
#no_actnorm|8|512|1e-3|0|0.0

# ---------- Presets ----------
# name|num_flows|num_hidden|lr|use_act_norm|wd
PRESETS="$(cat <<'EOF'
small_actnorm|4|256|1e-3|1|1e-4
deeper|12|384|1e-3|1|1e-4
wider|6|1024|1e-3|1|1e-4
large|12|768|1e-3|1|1e-4
larger|12|1024|5e-4|1|1e-4
xlarger|12|1280|5e-4|1|0.0
xlarger|12|1280|5e-4|1|1e-4
EOF
)"

# ---------- Loop & submit ----------
while IFS='|' read -r NAME NF NH LR ACT WD; do
  [[ -z "${NAME:-}" || "$NAME" =~ ^# ]] && continue
  EXP_NAME="nvp_${NAME}_${DS_TAG}"

  echo ">>> Submitting: ${EXP_NAME}  (ds=${DATASET}, epochs=${EPOCHS}, flows=${NF}, hidden=${NH}, lr=${LR}, actnorm=${ACT}, wd=${WD}) DEBUG:${IS_DEV}"
#  EXP_NAME="$EXP_NAME" \

  DATASET="$DATASET" \
  DS_TAG="$DS_TAG" \
  HV_DIM="$HV_DIM" \
  EPOCHS="$EPOCHS" \
  NUM_FLOWS="$NF" \
  NUM_HIDDEN_CHANNELS="$NH" \
  LR="$LR" \
  WEIGHT_DECAY="$WD" \
  USE_ACT_NORM="$ACT" \
  DEVICE="$DEVICE" \
  IS_DEV="$IS_DEV" \
  "$SUBMIT"

done <<< "$PRESETS"
