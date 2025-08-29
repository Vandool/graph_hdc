#!/bin/bash
#
# bwUniCluster 3.0 — single-GPU dev job
# Partitions:
# cpu_il dev_cpu_il | cpu dev_cpu | highmem dev_highmem | gpu_h100 dev_gpu_h100 | gpu_mi300 | gpu_a100_il gpu_h100_il|

#SBATCH --job-name=zinc_pairs_baseline_mlp
#SBATCH --partition=gpu_a100_il
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load devel/cuda/11.8

# ========== Paths ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/classification"
SCRIPT_NAME="2_base_line_mlp_mirror.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# Optional: print env
echo "Running ${SCRIPT}"
nvidia-smi || true

# Run (pixi must be on PATH)
pixi run python "$SCRIPT" \
  --project_dir "$PROJECT_DIR" \
  --epochs 2 \
  --batch_size 64 \
  --hv_dim 7744 \
  --vsa HRR \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --micro_bs 32 \
  --train_parents 20000 \
  --valid_parents 2000
