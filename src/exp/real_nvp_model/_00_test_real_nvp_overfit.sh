#!/bin/bash

# ========== Experiment Path Configuration ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/real_nvp_model"

# ========== W&B Setup ==========
ENTITY="arvand-kaveh-karlsruhe-institute-of-technology"
PROJECT="realnvp-hdc"
SWEEP_IDS=(
  "55h2zcvr" 
  "1z257i4c"
  )

# ========== Sweep Config ==========
HV_DIMS=(6400 9216)

# ========== Slurm Sweep Submission ==========
for idx in "${!HV_DIMS[@]}"; do
  hv_dim="${HV_DIMS[$idx]}"
  sweep_id="${SWEEP_IDS[$idx]}"

  sbatch \
    --job-name="sweep_hdc${hv_dim}" \
    --partition=dev_gpu_h100 \
    --time=00:10:00 \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=1 \
    --mem=16G \
    --wrap="module load devel/cuda/11.8 && \
      export WANDB_API_KEY=${WANDB_API_KEY} && \
      export WANDB_SWEEP=1 && \
      cd ${EXPERIMENTS_PATH} && \
      pixi run wandb agent ${ENTITY}/${PROJECT}/${sweep_id}"
done