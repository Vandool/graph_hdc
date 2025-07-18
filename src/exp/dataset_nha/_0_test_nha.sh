#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128000mb

PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/dataset_nha"


module load devel/cuda/11.8

pixi run python "${EXPERIMENTS_PATH}/0_test_nha.py"