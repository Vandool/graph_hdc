#!/bin/bash
#
# bwUniCluster 3.0 — single-GPU dev job
# Partitions:
# cpu_il dev_cpu_il | cpu dev_cpu | highmem dev_highmem | gpu_h100 dev_gpu_h100 | gpu_mi300 | gpu_a100_il gpu_h100_il|

#SBATCH --job-name=ZincPairsV2
#SBATCH --partition=cpu          # standard CPU partition (no GPUs)
#SBATCH --time=24:00:00          # reasonable walltime (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # 8 CPU cores for your Python job
#SBATCH --mem=64G                # ~32 GB RAM is a sensible default

module load devel/cuda/11.8

# ========== Experiment Path Configuration ==========
PROJECT_DIR="${GHDC_HOME}"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/dataset_generation"
SCRIPT_NAME="generate_zinc_pairs.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# (optional) ensure logs dir exists if you redirect elsewhere
# mkdir -p "${PROJECT_DIR}/logs"

# Your environment/conda/module loads would go here if needed
# module load ...

# Run (pixi must be on PATH in your environment)
pixi run python "$SCRIPT"