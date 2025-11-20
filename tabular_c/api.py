"""
Public programmatic API for the TACO (Tabular Corruptions) benchmark.
"""
from __future__ import annotations

# Config first to set env vars.
from . import config

from typing import Sequence, Dict, Any
import numpy as np
import pandas as pd
from .benchmark import run_classification_suite, run_regression_suite

def run_benchmark(
    dataset_name: str,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train,
    y_test,
    results_dir: str = config.DEFAULT_RESULTS_DIR,
    task: str | None = None,
    severities: Sequence[float] = config.DEFAULT_SEVERITIES,
) -> Dict[str, Any]:
    """
    Run the TACO benchmark on a single dataset programmatically.
    """
    if task is None:
        y_arr = np.asarray(y_train)
        if np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype == bool:
            task = config.TASK_CLASSIFICATION
        else:
            task = config.TASK_REGRESSION

    if task == config.TASK_CLASSIFICATION:
        run_classification_suite(dataset_name, X_train, X_test, y_train, y_test, results_dir, tuple(severities), config.RANDOM_STATE)
    else:
        run_regression_suite(dataset_name, X_train, X_test, y_train, y_test, results_dir, tuple(severities), config.RANDOM_STATE)

    return {
        "dataset_name": dataset_name,
        "task": task,
        "results_dir": results_dir,
        "severities": list(severities),
    }