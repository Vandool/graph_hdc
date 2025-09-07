#!/bin/bash
#
# bwUniCluster 3.0 — single-GPU job (simple, resume-friendly)

#> 1.	Small + ActNorm
# •	NUM_FLOWS=4, NUM_HIDDEN=256, LR=1e-3, USE_ACT_NORM=1

#  2.	Baseline
# •	NUM_FLOWS=8, NUM_HIDDEN=512, LR=1e-3, USE_ACT_NORM=1

# 3.	Baseline, lower LR
# •	NUM_FLOWS=8, NUM_HIDDEN=512, LR=5e-4, USE_ACT_NORM=1

# 4.	Deeper (depth test)
# •	NUM_FLOWS=12, NUM_HIDDEN=384, LR=1e-3, USE_ACT_NORM=1

# 5.	Wider (width test)
# •	NUM_FLOWS=6, NUM_HIDDEN=1024, LR=1e-3, USE_ACT_NORM=1

# 6.	ActNorm ablation
# •	NUM_FLOWS=8, NUM_HIDDEN=512, LR=1e-3, USE_ACT_NORM=0

# 7.	Large
# •	NUM_FLOWS=12, NUM_HIDDEN=768, LR=1e-3, USE_ACT_NORM=1


# ---------- tweak these ----------
ENTITY="arvand-kaveh-karlsruhe-institute-of-technology"
PROJECT="real_nvp_v3_dev"
EXPERIMENTS_PATH="${GHDC_HOME}/src/exp/real_nvp_v2"

# run mode
DEV=0                     # 1 = dev queue (00:10:00), 0 = normal (24:00:00)

# Leave empty to build a deterministic name from params below. Or set it to be used
EXP_NAME="real_nvp_v3_dev"

# resume from last checkpoint in this EXP_NAME (if it exists)
RESUME=0

# training/model config
EPOCHS=500
BATCH_SIZE=256
DEVICE="cuda"
HV_DIM=$((88*88))
NUM_FLOWS=8
NUM_HIDDEN=512
USE_ACT_NORM=1            # 1 = pass flag, 0 = don’t
LR=1e-4
WEIGHT_DECAY=0.0
SEED=42
# ---------------------------------
# Available partitions: 
# Available dev partitions: 

# Partitions and time limits
PARTITION_NORMAL="accelerated"      # cpuonly large accelerated accelerated-h100 accelerated-200 
PARTITION_DEV="dev_accelerated"     # ddev_cpuonly dev_accelerated dev_accelerated-h100
TIME_NORMAL="36:00:00"
TIME_DEV="00:10:00"

if [[ "$DEV" -eq 1 ]]; then
  PARTITION="$PARTITION_DEV"
  TIME="$TIME_DEV"
else
  PARTITION="$PARTITION_NORMAL"
  TIME="$TIME_NORMAL"
fi

# --- deterministic experiment name (only if EXP_NAME not provided) ---
# keep only safe filename chars for numeric tokens
to_token () { echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'; }

if [[ -z "$EXP_NAME" ]]; then
  LR_TOK=$(to_token "$LR")
  WD_TOK=$(to_token "$WEIGHT_DECAY")
  AN_TOK=$([ "$USE_ACT_NORM" -eq 1 ] && echo "an" || echo "noan")
  # exclude EPOCHS on purpose so you can resume the same folder with different epoch budgets
  EXP_NAME="h${HV_DIM}_f${NUM_FLOWS}_hid${NUM_HIDDEN}_s${SEED}_lr${LR_TOK}_wd${WD_TOK}_${AN_TOK}"
fi

# Build python args
PY_ARGS=( "0_real_nvp_v2.py"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --device "$DEVICE"
  --hv_dim "$HV_DIM"
  --num_flows "$NUM_FLOWS"
  --num_hidden_channels "$NUM_HIDDEN"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --exp_dir_name "$EXP_NAME"
)

# act-norm flag (only pass when true; argparse bool with type=bool treats any string as True)
if [[ "$USE_ACT_NORM" -eq 1 ]]; then
  PY_ARGS+=( --use_act_norm )
fi

# optional resume from last.ckpt if present
CKPT_PATH="${EXPERIMENTS_PATH}/results/0_real_nvp_v2/${EXP_NAME}/models/last.ckpt"
if [[ "$RESUME" -eq 1 && -f "$CKPT_PATH" ]]; then
  PY_ARGS+=( --continue_from "$CKPT_PATH" )
fi

JOB_NAME="2RealNVP_${EXP_NAME}"

sbatch \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks=1 \
  --mem=64G \
  --wrap="module load devel/cuda && \
    export WANDB_API_KEY=\${WANDB_API_KEY} && \
    export WANDB_ENTITY='${ENTITY}' && \
    export WANDB_PROJECT='${PROJECT}' && \
    export WANDB_NAME='${EXP_NAME}' && \
    cd '${EXPERIMENTS_PATH}' && \
    pixi run python ${PY_ARGS[*]}"