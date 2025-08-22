#!/bin/bash
#SBATCH --job-name=hello_slurm
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --output=optuna_%A_%a.out
#SBATCH --error=optuna_%A_%a.err
#SBATCH --array=1-5
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

# ========== Experiment Path Configuration ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/optuna"
SCRIPT_NAME="hello_slurm.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"


pixi run python "${SCRIPT}"