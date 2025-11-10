#!/bin/bash
# Submit property regression HPO for ZINC_SMILES_HRR_1024_F64_5G1NG4
# All properties: logp, sa_score, qed, max_ring_size
# 100 trials each

set -euo pipefail

DATASET="QM9_SMILES_HRR_256_F64_G1NG3"
N_TRIALS=20
CLUSTER="${CLUSTER:-uc3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting Property Regression HPO"
echo "Dataset: ${DATASET}"
echo "Trials per property: ${N_TRIALS}"
echo "Cluster: ${CLUSTER}"
echo "=========================================="

# Array of all properties
# PROPERTIES=("logp" "sa_score" "qed" "max_ring_size")
PROPERTIES=("sa_score" "qed" "max_ring_size")

for PROPERTY in "${PROPERTIES[@]}"; do
    echo ""
    echo ">>> Submitting: ${PROPERTY} on ${DATASET}"

    DATASET="${DATASET}" \
    N_TRIALS="${N_TRIALS}" \
    PROPERTY="${PROPERTY}" \
    CLUSTER="${CLUSTER}" \
    bash "${SCRIPT_DIR}/_2_pr_submit.sh"

    echo "✓ Submitted: ${PROPERTY}"

    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "=========================================="
echo "All jobs submitted for ${DATASET}!"
echo "Total: ${#PROPERTIES[@]} properties × ${N_TRIALS} trials"
echo "=========================================="
