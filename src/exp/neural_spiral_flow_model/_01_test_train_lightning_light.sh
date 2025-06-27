#!/bin/bash

# ========== Experiment Path Configuration ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/neural_spiral_flow_model"
SCRIPT="${EXPERIMENTS_PATH}/01_test_train_ligthning_light.py"  # Update to your actual script!

# ========== General Config ==========
BASE_DIR="${PROJECT_DIR}/artefacts"

# ========== HDC Config ==========
VSA_MODELS=("HRR" "MAP")
HV_DIMS=(6400 9216)           # 80*80=6400, 96*96=9216
DATASET="ZINC_ND_COMB"

# ========== Spiral Flow Config ==========
NUM_INPUT_CHANNELS=(19200 27648)  # 3*6400=19200, 3*9216=27648
NUM_FLOWS=8
NUM_BLOCKS=2
NUM_HIDDEN_CHANNELS=128
NUM_CONTEXT_CHANNELS=0           # or set to None if not used
NUM_BINS=8
TAIL_BOUND=3
ACTIVATION="relu"                # relu, gelu, leakyrelu
DROPOUT_PROBABILITY=0.0
PERMUTE=false                    # pass --permute if true
INIT_IDENTITY=true               # pass --init_identity if true
INPUT_SHAPE="3,6400"             # or "3,9216"
LR=0.001
WEIGHT_DECAY=0.0
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
          --base_dir ${BASE_DIR} \\
          --vsa_model ${vsa} \\
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
          --dropout_probability ${DROPOUT_PROBABILITY} \\
          $([[ $PERMUTE == true ]] && echo '--permute') \\
          $([[ $INIT_IDENTITY == true ]] && echo '--init_identity') \\
          --input_shape ${input_shape} \\
          --lr ${LR} \\
          --weight_decay ${WEIGHT_DECAY} \\
          --device ${DEVICE} \\
          "
  done
done