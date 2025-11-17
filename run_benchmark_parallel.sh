#!/bin/bash
# run_benchmark_parallel.sh
# --------------------------------------------------------------------------------
# Purpose: Runs the TACO benchmark across all specified datasets in parallel
#          to dramatically reduce total execution time.
#
# Usage: ./run_benchmark_parallel.sh [DATA_DIR] [RESULTS_DIR] [ONLY_DATASET_LIST]
#        Example: ./run_benchmark_parallel.sh data results diabetes,porto,ieee,nyc
# --------------------------------------------------------------------------------

# Configuration
DATA_DIR="${1:-data}"        # Defaults to 'data' if not provided
RESULTS_DIR="${2:-results}"  # Defaults to 'results' if not provided
# List of datasets from config.py/cli.py. Use 'all' if no list is provided.
DATASETS_LIST="${3:-adult,diabetes,porto,ieee,nyc}"

# Convert comma-separated string to an array
IFS=',' read -r -a DATASETS <<< "$DATASETS_LIST"

# This ensures that the script runs from the directory where 'tabular-c' is located.
# ${BASH_SOURCE[0]} is the path to the currently executing script.
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJECT_ROOT=$(cd "$SCRIPT_DIR" && pwd)

echo "======================================================"
echo "TACO Benchmark Parallel Runner"
echo "Project Root (Current Directory for jobs): $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Datasets to run: ${DATASETS[*]}"
echo "======================================================"

# Array to store the Process IDs (PIDs) of background jobs
PIDS=()

# --- Start Parallel Jobs ---
for dataset in "${DATASETS[@]}"; do
    echo "[LAUNCHING] Starting benchmark for dataset: $dataset..."

    (
        cd "$PROJECT_ROOT" || exit
        python -m tabular_c.cli \
            --only "$dataset" \
            --data_dir "$DATA_DIR" \
            --results_dir "$RESULTS_DIR" \
            > "$RESULTS_DIR/${dataset}_log.txt" 2>&1
    ) &

    # Save the Process ID for later waiting
    PIDS+=($!)
done

echo "------------------------------------------------------"
echo "All ${#DATASETS[@]} jobs launched in the background. Waiting for completion..."
echo "------------------------------------------------------"


# --- Wait for All Jobs to Complete ---
SUCCESS_COUNT=0
FAILURE_COUNT=0

for pid in "${PIDS[@]}"; do
    # Wait for a specific process ID
    # Note: wait "$pid" must be called outside the subshell ( )
    if wait "$pid"; then
        echo "[SUCCESS] Process $pid finished successfully."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "[FAILURE] Process $pid exited with an error. Check logs in $RESULTS_DIR."
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi
done


# --- Final Summary ---
echo "======================================================"
echo "PARALLEL BENCHMARK COMPLETE"
echo "Successful runs: $SUCCESS_COUNT"
echo "Failed runs: $FAILURE_COUNT"
echo "Results written to $RESULTS_DIR/*.csv"
echo "Individual logs available in $RESULTS_DIR/*.txt"
echo "======================================================"