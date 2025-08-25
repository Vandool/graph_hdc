#!/bin/bash
ENTITY="arvand-kaveh-karlsruhe-institute-of-technology"
PROJECT="realnvp-hdc"
SWEEP_YAML="sweep_7744_test.yml"
EXPERIMENTS_PATH="${GHDC_HOME}/src/exp/real_nvp_model"

# 1) Create sweep once and capture its ID
# shellcheck disable=SC2164
cd "${EXPERIMENTS_PATH}"
SWEEP_ID=$(pixi run wandb sweep "${SWEEP_YAML}" --entity "${ENTITY}" --project "${PROJECT}" | awk '/Created sweep with ID/ {print $NF}')

# 2) Submit agent
sbatch \
  --job-name="sweep_hdc7744" \
  --partition=dev_gpu_h100 \
  --time=00:20:00 \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks=1 \
  --mem=16G \
  --wrap="module load devel/cuda/11.8 && \
    export WANDB_API_KEY=${WANDB_API_KEY} && \
    export WANDB_SWEEP=1 && \
    cd ${EXPERIMENTS_PATH} && \
    pixi run wandb agent ${ENTITY}/${PROJECT}/${SWEEP_ID}"