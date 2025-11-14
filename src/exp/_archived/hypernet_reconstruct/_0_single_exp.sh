#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000mb

PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/graph_hdc/experiments/hypernet_reconstruct"


DATASET="ZINC_ND_COMB"
SEED="42"
DATA_BATCH_SIZE="32"
VSA="HRR"
HV_DIM="9216"
HYPERNET_DEPTH="3"
MAX_MAIN_LOOP="3"
REC_NUM_ITER="10"
REC_LEARNING_RATE="1"
REC_BATCH_SIZE="1"
REC_LOW="10"
REC_HIGH="10"
REC_ALPHA="0.1"
REC_LAMBDA_L1="0.01"


module load devel/cuda/11.8

pixi run python "${EXPERIMENTS_PATH}/01_hypernet_reconstruct_single_exp.py" \
                          --PROJECT_DIR="\"${PROJECT_DIR}\"" \
                          --SEED="${SEED}" \
                          --DATASET="\"${DATASET}\"" \
                          --DATA_BATCH_SIZE="${DATA_BATCH_SIZE}" \
                          --VSA="\"${VSA}\"" \
                          --HV_DIM="${HV_DIM}" \
                          --HYPERNET_DEPTH="${HYPERNET_DEPTH}" \
                          --MAX_MAIN_LOOP="${MAX_MAIN_LOOP}" \
                          --REC_NUM_ITER="${REC_NUM_ITER}" \
                          --REC_LEARNING_RATE="${REC_LEARNING_RATE}" \
                          --REC_BATCH_SIZE="${REC_BATCH_SIZE}" \
                          --REC_LOW="${REC_LOW}" \
                          --REC_HIGH="${REC_HIGH}" \
                          --REC_ALPHA="${REC_ALPHA}" \
                          --REC_LAMBDA_L1="${REC_LAMBDA_L1}"