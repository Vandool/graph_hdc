#!/bin/bash

# ========== Experiment Path Configuration ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/neural_spiral_flow_model"
SCRIPT_NAME="04_test_train_ligthning_light_small_batch.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# ========== General Config ==========
BASE_DIR="${EXPERIMENTS_PATH}/results/${SCRIPT_NAME%.*}"  # Remove .py extension for directory name

# ========== HDC Config ==========
VSA_MODELS=("HRR")
HV_DIMS=(1024)           # 80*80=6400, 96*96=9216
DATASET="ZINC_ND_COMB"
SEED=42
EPOCHS=100
BATCH_SIZE=16

# ========== Spiral Flow Config ==========
NUM_INPUT_CHANNELS=(3072)  # 3*6400=19200, 3*9216=27648
NUM_FLOWS=16
NUM_BLOCKS=2
NUM_HIDDEN_CHANNELS=128
NUM_CONTEXT_CHANNELS=0           # or set to None if not used
NUM_BINS=8
TAIL_BOUND=3
FLOW_TYPE="autoregressive"
ACTIVATION="leakyrelu"                # relu, gelu, leakyrelu
DROPOUT_PROBABILITY=0.0
PERMUTE=false                    # pass --permute if true
INIT_IDENTITY=true               # pass --init_identity if true
INPUT_SHAPE="3,1024"             # or "3,9216"
LR=0.001                         # 1e-3 
WEIGHT_DECAY=0.00001             # 1e-5
DEVICE="cuda"


for idx in "${!HV_DIMS[@]}"; do
  hv_dim="${HV_DIMS[$idx]}"
  num_input_channels="${NUM_INPUT_CHANNELS[$idx]}"
  input_shape="3,${hv_dim}"

  for vsa in "${VSA_MODELS[@]}"; do
    sbatch \
      --job-name=spiralflow \
      --partition=dev_gpu_h100 \
      --time=00:30:00 \
      --gres=gpu:1 \
      --nodes=1 \
      --ntasks=1 \
      --mem=16G \
      --wrap="module load devel/cuda/11.8 && \\
        pixi run python ${SCRIPT} \\
          --project_dir ${PROJECT_DIR} \\
          --base_dir ${BASE_DIR} \\
          --vsa ${vsa} \\
          --hv_dim ${hv_dim} \\
          --dataset ${DATASET} \\
          --num_input_channels ${num_input_channels} \\
          --num_flows ${NUM_FLOWS} \\
          --num_blocks ${NUM_BLOCKS} \\
          --num_hidden_channels ${NUM_HIDDEN_CHANNELS} \\
          --num_context_channels ${NUM_CONTEXT_CHANNELS} \\
          --num_bins ${NUM_BINS} \\
          --tail_bound ${TAIL_BOUND} \\
          --activation ${ACTIVATION} \\
          --flow_type ${FLOW_TYPE} \\
          --dropout_probability ${DROPOUT_PROBABILITY} \\
          $([[ $PERMUTE == true ]] && echo '--permute') \\
          $([[ $INIT_IDENTITY == true ]] && echo '--init_identity') \\
          --input_shape ${input_shape} \\
          --lr ${LR} \\
          --weight_decay ${WEIGHT_DECAY} \\
          --device ${DEVICE} \\
          --seed ${SEED} \\
          --epochs ${EPOCHS} \\
          --batch_size ${BATCH_SIZE} \\
          "
  done
done