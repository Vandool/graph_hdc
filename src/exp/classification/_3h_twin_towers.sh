#!/bin/bash
#
# HoreKa — single-GPU dev job (A100 or H100)
# Available partitions: cpuonly large accelerated accelerated-h100 accelerated-200 
# Available dev partitions: dev_cpuonly dev_accelerated dev_accelerated-h100

#SBATCH --job-name=zinc_pairs_baseline_mlp
#SBATCH --partition=dev_accelerated
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# CUDA toolchain on HoreKa (generic “devel/cuda” is recommended)
module load devel/cuda
# If you must pin a version, check availability first:
# module avail devel/cuda

# ========== Paths ==========
PROJECT_DIR="${GHDC_HOME}"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/classification"
SCRIPT_NAME="3_twin_towers.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"


echo "Running ${SCRIPT} on ${SLURM_JOB_PARTITION} (${SLURM_CLUSTER_NAME})"
nvidia-smi || true

# Run (pixi must be on PATH)
pixi run python "$SCRIPT" \
  --project_dir "$PROJECT_DIR" \
  --epochs 10 \
  --batch_size 128 \
  --hv_dim 7744 \
  --vsa HRR \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --micro_bs 64 \
  --train_parents 100 \
  --valid_parents 100
