#!/bin/bash
#!/usr/bin/env bash
# Master script to submit all retrieval experiments to cluster
# This script submits individual jobs for each experiment configuration.
#
# Usage:
#   bash submit_all_jobs.sh
#
# Environment variables:
#   N_SAMPLES: Number of samples per experiment (default: 1000)
#   OUTPUT_DIR: Base output directory (default: ./results)
#   CLUSTER: Cluster name (uc3|hk|haic|local) (default: uc3)
#   ONLY_PARTITIONS: Comma-separated partition filter (optional)
#   DRY_RUN: Set to 1 to see commands without submitting (default: 0)

set -euo pipefail

# Configuration
export N_SAMPLES="${N_SAMPLES:-1000}"
export OUTPUT_DIR="${OUTPUT_DIR:-./results}"
export CLUSTER="${CLUSTER:-uc3}"
export DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting Retrieval Experiments to Cluster"
echo "=========================================="
echo "N_SAMPLES       : $N_SAMPLES"
echo "OUTPUT_DIR      : $OUTPUT_DIR"
echo "CLUSTER         : $CLUSTER"
echo "DRY_RUN         : $DRY_RUN"
echo "ONLY_PARTITIONS : ${ONLY_PARTITIONS:-all}"
echo "=========================================="

# Define experiment parameters
VSA_MODELS=("HRR" "MAP")
HV_DIMS=(256 512 1024 1600 2048)
DEPTHS=(2 3 4 5)

# Counter for submitted jobs
TOTAL_JOBS=0

# QM9 experiments (iteration budget = 1)
echo ""
echo "Submitting QM9 experiments..."
echo ""

for vsa in "${VSA_MODELS[@]}"; do
    for dim in "${HV_DIMS[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            echo ">>> Submitting QM9 - VSA=$vsa, dim=$dim, depth=$depth, iter_budget=1"
            bash "$SCRIPT_DIR/submit_single_job.sh" "$vsa" "$dim" "$depth" qm9 1
            ((TOTAL_JOBS++))
            # Small delay to avoid overwhelming the scheduler
            sleep 0.2
        done
    done
done

# ZINC experiments (iteration budgets = 1, 10, 20)
echo ""
echo "Submitting ZINC experiments..."
echo ""

ITER_BUDGETS=(1 10 20)

for vsa in "${VSA_MODELS[@]}"; do
    for dim in "${HV_DIMS[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            for iter_budget in "${ITER_BUDGETS[@]}"; do
                echo ">>> Submitting ZINC - VSA=$vsa, dim=$dim, depth=$depth, iter_budget=$iter_budget"
                bash "$SCRIPT_DIR/submit_single_job.sh" "$vsa" "$dim" "$depth" zinc "$iter_budget"
                ((TOTAL_JOBS++))
                # Small delay to avoid overwhelming the scheduler
                sleep 0.2
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Total configurations submitted: $TOTAL_JOBS"
echo "=========================================="
echo ""
echo "Monitor job status with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER | tail -20'"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo ""
echo "After jobs complete, generate plots with:"
echo "  cd $SCRIPT_DIR"
echo "  python plot_results.py --results_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR/plots"
echo ""
echo "To cancel all submitted jobs:"
echo "  scancel -u \$USER -n Retrieval_*"
