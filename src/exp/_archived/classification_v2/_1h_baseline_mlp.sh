#!/bin/bash
#
# HoreKa — single-GPU dev job (A100 or H100)
# Available partitions: cpuonly large accelerated accelerated-h100 accelerated-200 
# Available dev partitions: dev_cpuonly dev_accelerated dev_accelerated-h100

#SBATCH --job-name=1MLP_BASE_LayNorm
#SBATCH --partition=accelerated
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# CUDA toolchain on HoreKa (generic “devel/cuda” is recommended)
module load devel/cuda
# If you must pin a version, check availability first:
# module avail devel/cuda

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
  --exp_dir_name "mlp_stratified_base" \
  --epochs 3 \
  --batch_size 256 \
  --hv_dim 7744 \
  --lr 1e-4 \
  --weight_decay 0 \
  --num_workers 0 \
  --micro_bs 64 \
  --save_every_seconds 3600 \
  --keep_last_k 2 \
  --stratify True \
  --p_per_parent 20 \
  --n_per_parent 20 \
  --oracle_beam_size 8 \
  --oracle_num_evals 4 \
  --resample_training_data_on_batch True \
  --use_batch_norm True \
  --use_layer_norm False
