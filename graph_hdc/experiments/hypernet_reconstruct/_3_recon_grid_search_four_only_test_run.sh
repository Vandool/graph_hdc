#!/bin/bash

PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/graph_hdc/experiments/hypernet_reconstruct"
SCRIPT="${EXPERIMENTS_PATH}/02_hypernet_reconstruct_multiple.py"

DATASET="ZINC_ND_COMB"
SEED="42"
DATA_BATCH_SIZE="32"
HYPERNET_DEPTH="3"
MAX_MAIN_LOOP="1"

# Grid search
hv_dims=(9216)  # 80*80=6400, 96*96=9216
VSAs=("HRR")


for hv_dim in "${hv_dims[@]}"; do
  for vsa in "${VSAs[@]}"; do
    sbatch \
      --job-name=hyperrec \
      --partition=dev_gpu_h100 \
      --time=00:30:00 \
      --gres=gpu:1 \
      --nodes=1 \
      --ntasks=1 \
      --mem=50G \
      --wrap="module load devel/cuda/11.8 && \\
        pixi run python ${SCRIPT} \\
          --PROJECT_DIR=\"\\\"${PROJECT_DIR}\\\"\" \\
          --SEED=\"${SEED}\" \\
          --DATASET=\"\\\"${DATASET}\\\"\" \\
          --DATA_BATCH_SIZE=\"${DATA_BATCH_SIZE}\" \\
          --VSA=\"\\\"${vsa}\\\"\" \\
          --HV_DIM=\"${hv_dim}\" \\
          --HYPERNET_DEPTH=\"${HYPERNET_DEPTH}\" \\
          --MAX_MAIN_LOOP=\"${MAX_MAIN_LOOP}\" \\
          "
  done
done