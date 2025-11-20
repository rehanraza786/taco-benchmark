#!/bin/bash
# run_benchmark_parallel.sh
# --------------------------------------------------------------------------------
# Purpose: Runs the TACO benchmark across all specified datasets in parallel.
#          Optimized for high-throughput execution on multi-core systems.
#
# Usage: ./run_benchmark_parallel.sh [DATA_DIR] [RESULTS_DIR] [DATASETS]
#        Example: ./run_benchmark_parallel.sh data results_v1 adult,diabetes
# --------------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status.

# Configuration
DATA_DIR="${1:-data}"
RESULTS_DIR="${2:-results}"
DATASETS_LIST="${3:-adult,diabetes,porto,ieee,nyc}"

# Convert comma-separated string to an array
IFS=',' read -r -a DATASETS <<< "$DATASETS_LIST"

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJECT_ROOT=$(cd "$SCRIPT_DIR" && pwd)

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

echo "======================================================"
echo "🌮 TACO Benchmark Parallel Runner"
echo "------------------------------------------------------"
echo "Project Root : $PROJECT_ROOT"
echo "Data Dir     : $DATA_DIR"
echo "Results Dir  : $RESULTS_DIR"
echo "Datasets     : ${DATASETS[*]}"
echo "======================================================"

PIDS=()

for dataset in "${DATASETS[@]}"; do
    echo "[LAUNCHING] $dataset..."

    # Run in background
    # We explicitly use python -m tabular_c.cli to ensure relative imports work
    (
        cd "$PROJECT_ROOT" || exit
        python -m tabular_c.cli \
            --only "$dataset" \
            --data_dir "$DATA_DIR" \
            --results_dir "$RESULTS_DIR" \
            > "$RESULTS_DIR/${dataset}_log.txt" 2>&1
    ) &

    PIDS+=($!)
done

echo "------------------------------------------------------"
echo "All jobs launched. Logs are being written to $RESULTS_DIR/"
echo "Waiting for completion..."
echo "------------------------------------------------------"

FAILURES=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "[DONE] Process $pid finished."
    else
        echo "[ERROR] Process $pid failed."
        FAILURES=$((FAILURES + 1))
    fi
done

echo "======================================================"
if [ $FAILURES -eq 0 ]; then
    echo "✅ BENCHMARK COMPLETE - ALL SUCCESS"
else
    echo "❌ BENCHMARK COMPLETE - $FAILURES FAILURES"
fi
echo "======================================================"