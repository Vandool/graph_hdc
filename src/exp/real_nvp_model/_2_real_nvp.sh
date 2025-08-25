#!/bin/bash
#
# bwUniCluster 3.0 — single-GPU dev job
# Partitions:
# cpu_il dev_cpu_il | cpu dev_cpu | highmem dev_highmem | gpu_h100 dev_gpu_h100 | gpu_mi300 | gpu_a100_il gpu_h100_il|

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
SWEEP_ID="z9im3p4w"

# 2) Submit agent
sbatch \
  --job-name="RealNVP" \
  --partition=gpu_h100_il \
  --time=24:00:00 \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks=1 \
  --mem=128G \
  --wrap="module load devel/cuda/11.8 && \
    export WANDB_API_KEY=${WANDB_API_KEY} && \
    export WANDB_SWEEP=1 && \
    cd ${EXPERIMENTS_PATH} && \
    pixi run wandb agent ${ENTITY}/${PROJECT}/${SWEEP_ID}"