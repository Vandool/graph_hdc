#!/bin/bash
#!/usr/bin/env bash
# Submit MAP experiments on ZINC dataset
# 60 jobs: 5 dimensions × 4 depths × 3 iteration_budgets
# Runtime: 3h, CPUs: 4
#
# Usage:
#   bash submit_map_zinc.sh
#
# Environment variables:
#   N_SAMPLES: Number of samples per experiment (default: 1000)
#   CLUSTER: Cluster name (uc3|hk|haic|local) (default: uc3)
#   DRY_RUN: Set to 1 to see commands without submitting (default: 0)

set -euo pipefail

# Configuration
export N_SAMPLES="${N_SAMPLES:-1000}"
export CLUSTER="${CLUSTER:-uc3}"
export DRY_RUN="${DRY_RUN:-0}"
export CPUS_PER_TASK=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting MAP ZINC Retrieval Experiments"
echo "=========================================="
echo "N_SAMPLES       : $N_SAMPLES"
echo "CLUSTER         : $CLUSTER"
echo "CPUs per task   : $CPUS_PER_TASK"
echo "DRY_RUN         : $DRY_RUN"
echo "=========================================="

# Experiment parameters
VSA="MAP"
DATASET="zinc"
HV_DIMS=(256 512 1024 1600 2048)
DEPTHS=(2 3 4 5)
ITER_BUDGETS=(1 10 20)

# Counter for submitted jobs
TOTAL_JOBS=0

echo ""
echo "Submitting MAP ZINC experiments..."
echo ""

for dim in "${HV_DIMS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        for iter_budget in "${ITER_BUDGETS[@]}"; do
            echo ">>> Submitting MAP ZINC - dim=$dim, depth=$depth, iter_budget=$iter_budget"

            # Override time for ZINC (3 hours)
            TIME_LIMIT="03:00:00" bash "$SCRIPT_DIR/submit_single_job.sh" "$VSA" "$dim" "$depth" "$DATASET" "$iter_budget"

            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            sleep 0.2
        done
    done
done

echo ""
echo "=========================================="
echo "MAP ZINC jobs submitted!"
echo "Total configurations submitted: $TOTAL_JOBS"
echo "=========================================="
