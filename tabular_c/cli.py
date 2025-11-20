"""
Command-line interface for the TACO benchmark.
"""
# Config must be imported FIRST to set threading env vars
# before numpy/pandas initialize their BLAS backends.
from . import config

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .benchmark import run_classification_suite, run_regression_suite
from .api import run_benchmark
from .datasets import (
    load_adult,
    load_diabetes_130,
    load_porto_seguro,
    load_ieee_cis,
    load_nyc_property_sales,
)
from .utils import infer_types

# --- DATASET_SPECS ---
DATASET_SPECS = [
    {
        "name": "adult",
        "loader": load_adult,
        "filename": config.DATASET_FILENAME_MAP["adult"],
        "task": config.TASK_CLASSIFICATION,
    },
    {
        "name": "diabetes",
        "loader": load_diabetes_130,
        "filename": config.DATASET_FILENAME_MAP["diabetes"],
        "task": config.TASK_CLASSIFICATION,
    },
    {
        "name": "porto",
        "loader": load_porto_seguro,
        "filename": config.DATASET_FILENAME_MAP["porto"],
        "task": config.TASK_CLASSIFICATION,
    },
    {
        "name": "ieee",
        "loader": load_ieee_cis,
        "filename": config.DATASET_FILENAME_MAP["ieee"],
        "task": config.TASK_CLASSIFICATION,
    },
    {
        "name": "nyc",
        "loader": load_nyc_property_sales,
        "filename": config.DATASET_FILENAME_MAP["nyc"],
        "task": config.TASK_REGRESSION,
    },
]
# ---------------------------------------------

def main():
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(description="TACO: Tabular Corruptions Benchmark CLI")
    ap.add_argument("--data_dir", default=config.DEFAULT_DATA_DIR, help="Path to data directory.")
    ap.add_argument("--results_dir", default=config.DEFAULT_RESULTS_DIR, help="Path for output files.")
    ap.add_argument(
        "--only",
        default="all",
        choices=["all"] + config.DATASET_NAMES,
        help="Limit to a subset of predefined datasets.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Use a single severity for faster sanity checks.",
    )

    # --- Arguments for Generic CSVs ---
    ap.add_argument("--csv", type=str, default=None, help="Path to a custom CSV file.")
    ap.add_argument("--target_col", type=str, default=None, help="Name of the target column.")
    ap.add_argument(
        "--task",
        type=str,
        default=None,
        choices=[config.TASK_CLASSIFICATION, config.TASK_REGRESSION, "both"],
        help="Specify the task type for custom CSVs."
    )

    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    severities = config.DEFAULT_SEVERITIES if not args.quick else config.QUICK_SEVERITIES

    # --- Handle Custom CSV Mode ---
    if args.csv and args.target_col:
        print(f"--- Running in Generic Mode on: {args.csv} ---")

        try:
            df = pd.read_csv(args.csv)
        except Exception as e:
            print(f"Error: Failed to load CSV '{args.csv}'. {e}")
            return

        if args.target_col not in df.columns:
            print(f"Error: Target column '{args.target_col}' not found.")
            return

        y = df[args.target_col]
        X = df.drop(columns=[args.target_col])
        X = infer_types(X)

        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype == bool:
            inferred_task = config.TASK_CLASSIFICATION
            stratify = y
        else:
            inferred_task = config.TASK_REGRESSION
            stratify = None

        tasks_to_run = [args.task] if args.task and args.task != "both" else ([config.TASK_CLASSIFICATION, config.TASK_REGRESSION] if args.task == "both" else [inferred_task])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify
        )

        dataset_name = os.path.basename(args.csv).replace(".csv", "")

        for task in tasks_to_run:
            print(f"\n--- Running task: {task} for {dataset_name} ---")
            run_benchmark(
                dataset_name=dataset_name,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                results_dir=args.results_dir, severities=severities, task=task
            )

    elif args.csv or args.target_col:
        print("Error: --csv and --target_col must be used together.")

    else:
        # --- Handle Predefined Datasets ---
        print("--- Running predefined TACO benchmarks ---")
        for spec in DATASET_SPECS:
            name = spec["name"]
            if args.only not in ("all", name): continue

            print(f"--- Running dataset: {name} ({spec['task']}) ---")
            path = os.path.join(args.data_dir, spec["filename"])

            try:
                X_train, X_test, y_train, y_test = spec["loader"](path=path, random_state=config.RANDOM_STATE)
            except FileNotFoundError as e:
                print(f"Skipping {name}: {e}")
                continue

            if spec["task"] == config.TASK_CLASSIFICATION:
                run_classification_suite(name, X_train, X_test, y_train, y_test, args.results_dir, severities, config.RANDOM_STATE)
            elif spec["task"] == config.TASK_REGRESSION:
                run_regression_suite(name, X_train, X_test, y_train, y_test, args.results_dir, severities, config.RANDOM_STATE)

        print("--- TACO Benchmark run complete. ---")

if __name__ == "__main__":
    main()