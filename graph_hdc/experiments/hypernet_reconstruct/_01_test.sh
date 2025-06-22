#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G

FOLDER_PATH="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${FOLDER_PATH}/graph_hdc/experiments/hypernet_reconstruct"
FILE_PATH="${FOLDER_PATH}/graph_hdc/experiments/hypernet_reconstruct/results/overwritten"


module load devel/cuda/11.8

pixi run python "${EXPERIMENTS_PATH}/01_test.py" --FILE_PATH="\"${FILE_PATH}\""