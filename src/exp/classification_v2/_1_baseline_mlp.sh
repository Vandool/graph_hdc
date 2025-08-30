#!/bin/bash
#
# bwUniCluster 3.0 — single-GPU dev job
# Partitions:
# cpu_il dev_cpu_il | cpu dev_cpu | highmem dev_highmem | gpu_h100 dev_gpu_h100 | gpu_mi300 | gpu_a100_il gpu_h100_il|

#SBATCH --job-name=MLP_test
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load devel/cuda/11.8

# ========== Paths ==========
PROJECT_DIR="${GHDC_HOME}"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/classification_v2"
SCRIPT_NAME="1_mlp.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# Optional: print env
echo "Running ${SCRIPT}"
nvidia-smi || true


#   --exp_dir_name "TEST" \
# Run (pixi must be on PATH)
pixi run python "$SCRIPT" \
  --project_dir "$PROJECT_DIR" \
  --exp_dir_name "TEST" \
  --epochs 100 \
  --batch_size 128 \
  --hv_dim 7744 \
  --vsa "HRR" \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --micro_bs 64 \
  --train_parents_start 0 \
  --train_parents_end 10 \
  --valid_parents_start 0 \
  --valid_parents_end 1 \
  --save_every_seconds 1800 \
  --keep_last_k 2