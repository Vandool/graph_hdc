#!/bin/bash
#SBATCH --job-name=ZincPairsV3
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2

PROJECT_DIR="${GHDC_HOME}"
SCRIPT="${PROJECT_DIR}/src/exp/dataset_generation/generate_zinc_pairs_v3.py"

SPLITS=(train valid test)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}


echo "[JOB] split=${SPLIT}"
pixi run -e cluster --frozen python "$SCRIPT" --split "$SPLIT"
