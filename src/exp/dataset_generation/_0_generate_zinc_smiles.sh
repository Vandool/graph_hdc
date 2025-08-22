#!/bin/bash
#
# bwUniCluster 3.0 — standard CPU node job
# Regular CPU queue: `cpu` (72h max). Dev queue `dev_cpu` is limited to ~30 min.  # docs
#

#SBATCH --job-name=zinc_smiles_with_hv
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G

module load devel/cuda/11.8

# ========== Experiment Path Configuration ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/dataset_generation"
SCRIPT_NAME="generate_zinc_smiles.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# (optional) ensure logs dir exists if you redirect elsewhere
# mkdir -p "${PROJECT_DIR}/logs"

# Your environment/conda/module loads would go here if needed
# module load ...

# Run (pixi must be on PATH in your environment)
pixi run python "$SCRIPT"