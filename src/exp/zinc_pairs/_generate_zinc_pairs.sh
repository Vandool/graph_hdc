#!/bin/bash
#
# bwUniCluster 3.0 — standard CPU node job
# Regular CPU queue: `cpu` (72h max). Dev queue `dev_cpu` is limited to ~30 min.  # docs
#

#SBATCH --job-name=zincpairs
#SBATCH --partition=cpu           # standard CPU partition (no GPUs)
#SBATCH --time=02:00:00           # reasonable walltime (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # 8 CPU cores for your Python job
#SBATCH --mem=64G                 # ~32 GB RAM is a sensible default
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ========== Experiment Path Configuration ==========
PROJECT_DIR="/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"
EXPERIMENTS_PATH="${PROJECT_DIR}/src/exp/zinc_pairs"
SCRIPT_NAME="generate_zinc_pairs.py"
SCRIPT="${EXPERIMENTS_PATH}/${SCRIPT_NAME}"

# (optional) ensure logs dir exists if you redirect elsewhere
# mkdir -p "${PROJECT_DIR}/logs"

# Your environment/conda/module loads would go here if needed
# module load ...

# Run (pixi must be on PATH in your environment)
pixi run python "$SCRIPT"