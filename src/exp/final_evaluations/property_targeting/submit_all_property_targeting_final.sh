#!/bin/bash

# Unified Property Targeting Final Evaluation Submission Script
# ===============================================================
#
# This script submits all 4 property-dataset combinations for final evaluation:
#   1. QM9 × LogP   (3 targets: -1.0, 0.5, 2.0)
#   2. QM9 × QED    (3 targets: 0.3, 0.45, 0.6)
#   3. ZINC × LogP  (3 targets: 2.0, 4.0, 6.0)
#   4. ZINC × QED   (3 targets: 0.6, 0.75, 0.9)
#
# Each job runs independently in parallel on the cluster.
# Total: 4 jobs × ~2-3 hours = wall time ~2-3 hours
#
# Usage:
#   # Quick verification (2 samples per target, ~5-10 min total)
#   N_SAMPLES=2 bash submit_all_property_targeting_final.sh
#
#   # Production run (10,000 samples per target, ~2-3 hours)
#   N_SAMPLES=10000 bash submit_all_property_targeting_final.sh
#
#   # With molecule drawings (adds ~30 min per job)
#   N_SAMPLES=10000 DRAW=1 MAX_DRAW=200 bash submit_all_property_targeting_final.sh
#
#   # Dry run to preview commands
#   DRY_RUN=1 N_SAMPLES=10000 bash submit_all_property_targeting_final.sh
#
#   # Specific cluster configuration
#   CLUSTER=uc3 ONLY_PARTITIONS=gpu_h100 N_SAMPLES=10000 bash submit_all_property_targeting_final.sh

# -----------------------------
# Configuration
# -----------------------------

# Number of samples per target (default: 10000 for production)
N_SAMPLES=${N_SAMPLES:-10000}

# Drawing options
DRAW=${DRAW:-0}
MAX_DRAW=${MAX_DRAW:-200}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-final_results}

# Dry run mode (preview commands without submitting)
DRY_RUN=${DRY_RUN:-0}

# Cluster configuration (passed through to submit script)
CLUSTER=${CLUSTER:-local}
ONLY_PARTITIONS=${ONLY_PARTITIONS:-}

# HPO results base directory
HPO_BASE_DIR="${HPO_BASE_DIR:-hpo_results_bw3}"

# -----------------------------
# HPO Directory Paths
# -----------------------------

# Define the 4 HPO result directories
# These should match your actual directory names from HPO runs
declare -A HPO_DIRS=(
    ["qm9_logp"]="logp_QM9_SMILES_HRR_256_F64_G1NG3_20251117_133407"
    ["qm9_qed"]="qed_QM9_SMILES_HRR_256_F64_G1NG3_20251117_132051"
    ["zinc_logp"]="logp_ZINC_SMILES_HRR_256_F64_5G1NG4_20251117_131847"
    ["zinc_qed"]="qed_ZINC_SMILES_HRR_256_F64_5G1NG4_20251117_132050"
)

# -----------------------------
# Validate HPO Directories
# -----------------------------

echo "====================================="
echo "Property Targeting Final Evaluation"
echo "====================================="
echo "N Samples per target: $N_SAMPLES"
echo "Draw molecules: $DRAW"
if [[ "$DRAW" == "1" ]]; then
    echo "Max molecules to draw: $MAX_DRAW"
fi
echo "Output directory: $OUTPUT_DIR"
echo "Cluster: $CLUSTER"
if [[ -n "$ONLY_PARTITIONS" ]]; then
    echo "Partition filter: $ONLY_PARTITIONS"
fi
echo "Dry run: $DRY_RUN"
echo "====================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if submit script exists
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_final_eval_parallel.sh"
if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
    echo "ERROR: Submission script not found: $SUBMIT_SCRIPT" >&2
    exit 1
fi

# Validate all HPO directories exist
echo "Validating HPO directories..."
MISSING_DIRS=()
for key in "${!HPO_DIRS[@]}"; do
    hpo_dir="${SCRIPT_DIR}/${HPO_BASE_DIR}/${HPO_DIRS[$key]}"
    if [[ ! -d "$hpo_dir" ]]; then
        echo "  ✗ Missing: $key -> $hpo_dir"
        MISSING_DIRS+=("$key")
    else
        echo "  ✓ Found: $key -> ${HPO_DIRS[$key]}"
    fi
done

if [[ ${#MISSING_DIRS[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: ${#MISSING_DIRS[@]} HPO directories not found:" >&2
    for key in "${MISSING_DIRS[@]}"; do
        echo "  - $key: ${HPO_BASE_DIR}/${HPO_DIRS[$key]}" >&2
    done
    echo ""
    echo "Please verify HPO_BASE_DIR and HPO directory names." >&2
    exit 1
fi

echo ""
echo "All HPO directories validated successfully!"
echo ""

# -----------------------------
# Submit Jobs
# -----------------------------

echo "====================================="
echo "Submitting 4 Jobs in Parallel"
echo "====================================="
echo ""

# Job submission counter
SUBMITTED=0
FAILED=0

# Submit each job
for key in qm9_logp qm9_qed zinc_logp zinc_qed; do
    hpo_dir="${SCRIPT_DIR}/${HPO_BASE_DIR}/${HPO_DIRS[$key]}"

    echo "----------------------------------------"
    echo "Job ${SUBMITTED}/4: $key"
    echo "----------------------------------------"
    echo "HPO Directory: ${HPO_DIRS[$key]}"
    echo ""

    # Build submission command
    export N_SAMPLES
    export DRAW
    export MAX_DRAW
    export OUTPUT_DIR
    export DRY_RUN
    export CLUSTER
    export ONLY_PARTITIONS

    # Submit the job
    if bash "$SUBMIT_SCRIPT" "$hpo_dir"; then
        ((SUBMITTED++))
        echo "  ✓ Submitted successfully"
    else
        ((FAILED++))
        echo "  ✗ Submission failed" >&2
    fi

    echo ""
done

# -----------------------------
# Summary
# -----------------------------

echo "====================================="
echo "Submission Summary"
echo "====================================="
echo "Jobs submitted: $SUBMITTED / 4"
if [[ $FAILED -gt 0 ]]; then
    echo "Jobs failed: $FAILED"
    echo ""
    echo "⚠️  Some jobs failed to submit. Check errors above."
    exit 1
else
    echo "Status: ✓ All jobs submitted successfully"
fi
echo "====================================="
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN COMPLETE - No jobs were actually submitted."
    echo ""
    echo "To submit for real, run without DRY_RUN:"
    echo "  N_SAMPLES=$N_SAMPLES bash $0"
    echo ""
else
    echo "Monitor job status with: squeue -u \$USER"
    echo ""
    echo "Expected completion time:"
    if [[ $N_SAMPLES -le 10 ]]; then
        echo "  ~5-10 minutes (verification mode)"
    elif [[ $N_SAMPLES -le 100 ]]; then
        echo "  ~15-30 minutes (quick test)"
    else
        echo "  ~2-3 hours (production mode)"
    fi
    echo ""
    echo "Results will be saved to: ${OUTPUT_DIR}/"
    echo ""

    # Show expected output structure
    echo "Expected output structure:"
    echo "  ${OUTPUT_DIR}/"
    for key in qm9_logp qm9_qed zinc_logp zinc_qed; do
        dataset_prop=$(echo "$key" | sed 's/_/ × /' | tr '[:lower:]' '[:upper:]')
        echo "  ├── ${HPO_DIRS[$key]%_*}_final_{timestamp}/"
        echo "  │   ├── aggregate_results.json"
        echo "  │   ├── target_X.X/"
        echo "  │   │   ├── results.json"
        echo "  │   │   ├── molecules_metadata.csv"
        echo "  │   │   ├── molecules.pkl"
        echo "  │   │   ├── plots/"
        echo "  │   │   └── drawings_all_valid/ (if DRAW=1)"
        echo "  │   └── plots/"
    done
    echo ""
fi
