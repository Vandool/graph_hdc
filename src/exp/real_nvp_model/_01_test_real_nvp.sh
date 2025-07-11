#!/bin/bash

# ========== Experiment Path Configuration ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/real_nvp_model"

# ========== W&B Setup ==========
export WANDB_API_KEY=49647f01af13168b83b51c3492e03c6f9a98ccd1
ENTITY="graph_hdc"
PROJECT="realnvp-hdc"
SWEEP_IDS=("abc123xyz" "def456uvw")  # Sweep IDs per hv_dim

# ========== Config ==========
HV_DIMS=(6400 9216)
VSA_MODELS=("HRR")

# ========== Slurm Sweep Submission ==========
for idx in "${!HV_DIMS[@]}"; do
  hv_dim="${HV_DIMS[$idx]}"
  sweep_id="${SWEEP_IDS[$idx]}"

  for vsa in "${VSA_MODELS[@]}"; do
    sbatch \
      --job-name="sweep_${vsa}_hdc${hv_dim}" \
      --partition=dev_gpu_h100 \
      --time=00:10:00 \
      --gres=gpu:1 \
      --nodes=1 \
      --ntasks=1 \
      --mem=16G \
      --wrap="module load devel/cuda/11.8 && \\
        export WANDB_SWEEP=1 && \\
        cd ${EXPERIMENTS_PATH} && \\
        pixi run wandb agent ${ENTITY}/${PROJECT}/${sweep_id} --vsa ${vsa} --hv_dim ${hv_dim}"
  done
done