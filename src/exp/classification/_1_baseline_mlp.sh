#!/bin/bash
#
# bwUniCluster 3.0 — single-GPU dev job

#SBATCH --job-name=zinc_pairs_baseline_mlp
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load devel/cuda/11.8

# ========== Paths ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/classification"
SCRIPT_NAME="1_base_line_mlp.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# Optional: print env
echo "Running ${SCRIPT}"
nvidia-smi || true

# Run (pixi must be on PATH)
pixi run python "$SCRIPT" \
  --project_dir "$PROJECT_DIR" \
  --epochs 20 \
  --batch_size 128 \
  --hv_dim 7744 \
  --vsa HRR \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --micro_bs 64 \
  --train_parents 50 \
  --valid_parents 10
