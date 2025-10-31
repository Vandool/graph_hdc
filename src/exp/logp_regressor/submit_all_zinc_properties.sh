#!/bin/bash
# Submit property regression HPO for BOTH ZINC datasets
# ZINC_SMILES_HRR_1024_F64_5G1NG4 and ZINC_SMILES_HRR_2048_F64_5G1NG4
# All properties: logp, sa_score, qed, max_ring_size
# 100 trials each

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-uc3}"

echo "=========================================="
echo "Submitting ALL ZINC Property Regression HPO"
echo "Datasets: ZINC_SMILES_HRR_1024_F64_5G1NG4"
echo "          ZINC_SMILES_HRR_2048_F64_5G1NG4"
echo "Properties: logp, sa_score, qed, max_ring_size"
echo "Trials per property: 100"
echo "Cluster: ${CLUSTER}"
echo "=========================================="

# Submit ZINC 1024
echo ""
echo "=== ZINC_SMILES_HRR_1024_F64_5G1NG4 ==="
CLUSTER="${CLUSTER}" bash "${SCRIPT_DIR}/submit_zinc1024_all_properties.sh"

echo ""
echo "Waiting 10 seconds before next dataset..."
sleep 10

# Submit ZINC 2048
echo ""
echo "=== ZINC_SMILES_HRR_2048_F64_5G1NG4 ==="
CLUSTER="${CLUSTER}" bash "${SCRIPT_DIR}/submit_zinc2048_all_properties.sh"

echo ""
echo "=========================================="
echo "✓ All submissions complete!"
echo "Total: 2 datasets × 4 properties × 100 trials = 800 trials"
echo "=========================================="
