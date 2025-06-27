#!/bin/bash

PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/graph_hdc/experiments/hypernet_reconstruct"
SCRIPT="${EXPERIMENTS_PATH}/01_hypernet_reconstruct_single_exp.py"

DATASET="ZINC_ND_COMB"
SEED="42"
DATA_BATCH_SIZE="32"
HYPERNET_DEPTH="3"
MAX_MAIN_LOOP="1"

# Grid search
hv_dims=(6400)  # 80*80=6400, 96*96=9216
VSAs=("HRR")
REC_NUM_ITERS=(10)
REC_LRS=(1.0 0.1)
REC_ALPHA=(0 0.1 0.5)
REC_LAMBDA_L1=(0 0.01 0.001 0.0001)

# Explicitly defined REC_LOW, REC_HIGH, REC_BATCH_SIZE combinations
rec_combinations=(
    "10 10 1"
    "-1 1 3"
    "-1 1 10"
    "-1 1 100"
    "0 3 3"
    "0 3 10"
    "0 3 100"
)

for hv_dim in "${hv_dims[@]}"; do
for vsa in "${VSAs[@]}"; do
for num_iter in "${REC_NUM_ITERS[@]}"; do
for lr in "${REC_LRS[@]}"; do
for alpha in "${REC_ALPHA[@]}"; do
for lambda in "${REC_LAMBDA_L1[@]}"; do
for combo in "${rec_combinations[@]}"; do
    read -r rec_low rec_high batch_size <<< "$combo"

    sbatch \
        --job-name=hyperrec \
        --partition=gpu_h100 \
        --time=00:30:00 \
        --gres=gpu:1 \
        --nodes=1 \
        --ntasks=1 \
        --mem=50G \
        --wrap="module load devel/cuda/11.8 && \
            pixi run python \"${SCRIPT}\" \
                --PROJECT_DIR=\"\\\"${PROJECT_DIR}\\\"\" \
                --SEED=${SEED} \
                --DATASET=\"\\\"${DATASET}\\\"\" \
                --DATA_BATCH_SIZE=${DATA_BATCH_SIZE} \
                --VSA=\"\\\"${vsa}\\\"\" \
                --HV_DIM=${hv_dim} \
                --HYPERNET_DEPTH=${HYPERNET_DEPTH} \
                --MAX_MAIN_LOOP=${MAX_MAIN_LOOP} \
                --REC_NUM_ITER=${num_iter} \
                --REC_LEARNING_RATE=${lr} \
                --REC_BATCH_SIZE=${batch_size} \
                --REC_LOW=${rec_low} \
                --REC_HIGH=${rec_high} \
                --REC_ALPHA=${alpha} \
                --REC_LAMBDA_L1=${lambda}"
done
done
done
done
done
done
done