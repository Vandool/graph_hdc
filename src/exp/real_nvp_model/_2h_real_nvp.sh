#!/bin/bash
#
# HoreKa — single-GPU dev job (A100 or H100)
# Available partitions: cpuonly large accelerated accelerated-h100 accelerated-200 
# Available dev partitions: dev_cpuonly dev_accelerated dev_accelerated-h100

ENTITY="arvand-kaveh-karlsruhe-institute-of-technology"
PROJECT="realnvp"
SWEEP_YAML="sweep_7744.yml"
echo ${GHDC_HOME}
EXPERIMENTS_PATH="${GHDC_HOME}/src/exp/real_nvp_model"

# 1) Create sweep once and capture its ID
# shellcheck disable=SC2164
# ========== W&B Setup ==========
cd "${EXPERIMENTS_PATH}"
## run: pixi run wandb sweep sweep_7748.yaml --entity "arvand-kaveh-karlsruhe-institute-of-technology" --project "realnvp"
SWEEP_ID="wl5c1put"

# 2) Submit agent
sbatch \
  --job-name="RealNVPHoreka" \
  --partition=accelerated-h100 \
  --time=24:00:00 \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks=1 \
  --mem=128G \
  --wrap="module load devel/cuda && \
    export WANDB_API_KEY=${WANDB_API_KEY} && \
    export WANDB_SWEEP=1 && \
    cd ${EXPERIMENTS_PATH} && \
    pixi run wandb agent ${ENTITY}/${PROJECT}/${SWEEP_ID}"