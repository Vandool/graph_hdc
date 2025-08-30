#!/bin/bash
#
# bwUniCluster 3.0 — single-GPU dev job
# Partitions:
# cpu_il dev_cpu_il | cpu dev_cpu | highmem dev_highmem | gpu_h100 dev_gpu_h100 | gpu_mi300 | gpu_a100_il gpu_h100_il|

#SBATCH --job-name=MLP
#SBATCH --partition=gpu_h100_il
#SBATCH --time=24:00:00
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
#  --continue_from "/home/ka/ka_iti/ka_zi9629/projects/graph_hdc/src/exp/classification_v2/results/1_mlp/TEST/models/last.pt"
# Run (pixi must be on PATH)
pixi run python "$SCRIPT" \
  --project_dir "$PROJECT_DIR" \
  --exp_dir_name "defaults" \
  --epochs 1 \
  --batch_size 128 \
  --hv_dim 7744 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --micro_bs 64 \
  --train_parents_start 0 \
  --train_parents_end 4000000 \
  --valid_parents_start 0 \
  --valid_parents_end 100000 \
  --save_every_seconds 1800 \
  --keep_last_k 2 
