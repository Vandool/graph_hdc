#!/bin/bash
#!/usr/bin/env bash
# Submit HRR experiments on ZINC dataset
# 48 jobs: 36 (pattern_matching: 3 dims × 4 depths × 3 iters) + 12 (greedy: 3 dims × 4 depths × 1)
# Runtime: 3h, CPUs: 4
#
# Usage:
#   bash submit_hrr_zinc.sh
#   BEAM_SIZE=128 bash submit_hrr_zinc.sh  # Custom beam size
#
# Environment variables:
#   N_SAMPLES: Number of samples per experiment (default: 1000)
#   CLUSTER: Cluster name (uc3|hk|haic|local) (default: uc3)
#   BEAM_SIZE: Beam size for greedy decoder (default: 64)
#   DRY_RUN: Set to 1 to see commands without submitting (default: 0)

set -euo pipefail

# Configuration
export N_SAMPLES="${N_SAMPLES:-200}"
export CLUSTER="${CLUSTER:-uc3}"
export DRY_RUN="${DRY_RUN:-0}"
export CPUS_PER_TASK=1
export BEAM_SIZE="${BEAM_SIZE:-64}"  # Default: 64, can be overridden via environment variable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting HRR ZINC Retrieval Experiments"
echo "=========================================="
echo "N_SAMPLES       : $N_SAMPLES"
echo "CLUSTER         : $CLUSTER"
echo "CPUs per task   : $CPUS_PER_TASK"
echo "BEAM_SIZE       : $BEAM_SIZE"
echo "DRY_RUN         : $DRY_RUN"
echo "=========================================="

# Experiment parameters
VSA="HRR"
DATASET="zinc"
#HV_DIMS=(256 512 1024)
HV_DIMS=(196)
DEPTHS=(4)
#ITER_BUDGETS=(1 10)
ITER_BUDGETS=(25)

# Counter for submitted jobs
TOTAL_JOBS=0

echo ""
echo "Submitting HRR ZINC experiments..."
echo ""

for dim in "${HV_DIMS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        # Pattern matching decoder: iterate over all iter_budgets
        for iter_budget in "${ITER_BUDGETS[@]}"; do
            echo ">>> Submitting HRR ZINC - dim=$dim, depth=$depth, iter_budget=$iter_budget, decoder=pattern_matching"
            TIME_LIMIT="10:00:00" bash "$SCRIPT_DIR/submit_single_job.sh" "$VSA" "$dim" "$depth" "$DATASET" "$iter_budget" "pattern_matching"
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            sleep 0.2
        done

        # Greedy decoder: only run once (iter_budget is irrelevant)
        echo ">>> Submitting HRR ZINC - dim=$dim, depth=$depth, iter_budget=1 (unused), decoder=greedy, beam_size=$BEAM_SIZE"
        TIME_LIMIT="04:00:00" bash "$SCRIPT_DIR/submit_single_job.sh" "$VSA" "$dim" "$depth" "$DATASET" "1" "greedy" "$BEAM_SIZE"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        sleep 0.2
    done
done

echo ""
echo "=========================================="
echo "HRR ZINC jobs submitted!"
echo "Total configurations submitted: $TOTAL_JOBS"
echo "=========================================="
