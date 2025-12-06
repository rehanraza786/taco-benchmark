from . import config

import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump, load
import tempfile
import shutil
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import gc

from .corruptions import CORRUPTION_FUNCS
from .models import (
    build_preprocessor,
    build_classifiers,
    build_regressors,
    evaluate_classifier,
    evaluate_regressor,
)
from .plots import plot_degradation
from .utils import ensure_dir


def _get_eval_set(X, y, limit, random_state, stratify=None):
    """
    Subsamples the dataset for evaluation.
    Uses numpy random choice for non-stratified cases.
    """
    n_samples = len(X)
    if n_samples <= limit:
        return X, y

    if stratify is not None:
        return resample(X, y, n_samples=limit, random_state=random_state, stratify=stratify, replace=False)

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_samples, size=limit, replace=False)

    if hasattr(X, "iloc"):
        X_sub = X.iloc[indices]
        y_sub = y.iloc[indices] if hasattr(y, "iloc") else y[indices]
    else:
        X_sub = X[indices]
        y_sub = y[indices]

    return X_sub, y_sub


def _evaluate_group_task(pipes_dict, X_eval_mmap, y_eval_mmap, corr_name, s, corr_fn,
                         task_seed, metric_name, is_classification,
                         col_indices, secondary_corr=None, precomputed_stds=None):
    """
    Worker function.
    """
    # Apply Primary Corruption
    # Use task-specific seed to ensure distinct entropy between corruptions
    Xc, yc = corr_fn(X_eval_mmap, y_eval_mmap, s, rng=task_seed, inplace=False,
                     col_idx=col_indices, precomputed_stds=precomputed_stds)

    final_corr_name = corr_name
    final_severity = s

    # Apply Secondary Corruption
    if secondary_corr:
        c2_name, c2_fn, s2 = secondary_corr
        Xc, yc = c2_fn(Xc, yc, s2, rng=task_seed, inplace=True,
                       col_idx=col_indices, precomputed_stds=precomputed_stds)
        final_corr_name = f"{corr_name}+{c2_name}"
        final_severity = f"{s:.1f}+{s2:.1f}"

    results = []

    # Evaluate All Models
    for model_name, pipe in pipes_dict.items():
        try:
            if is_classification:
                score, _ = evaluate_classifier(pipe, None, None, Xc, yc, refit=False)
            else:
                score, _ = evaluate_regressor(pipe, None, None, Xc, y_eval_mmap, refit=False)

            results.append([
                config.TASK_CLASSIFICATION if is_classification else config.TASK_REGRESSION,
                final_corr_name,
                final_severity,
                model_name,
                metric_name,
                score
            ])
        except Exception as e:
            print(f"Warning: Model {model_name} failed on {final_corr_name}: {e}")
            results.append([
                config.TASK_CLASSIFICATION if is_classification else config.TASK_REGRESSION,
                final_corr_name,
                final_severity,
                model_name,
                metric_name,
                np.nan
            ])

    # Explicit deletion helps reference counting, but we skip full GC for speed
    del Xc
    del yc

    return results


def run_suite_generic(name, X_train, X_test, y_train, y_test, results_dir, severities,
                      task_type, build_models_fn, metric_name, eval_fn):
    ensure_dir(results_dir)
    stratify = y_test if task_type == config.TASK_CLASSIFICATION else None

    X_eval, y_eval = _get_eval_set(X_test, y_test, config.EVAL_SAMPLE_LIMIT, config.RANDOM_STATE, stratify=stratify)

    num_cols = X_eval.select_dtypes(include=['float32', 'float64', 'int64', 'int32', 'int8']).columns
    cat_cols = X_eval.select_dtypes(include=['category']).columns

    col_indices = {
        'num': [X_eval.columns.get_loc(c) for c in num_cols],
        'cat': [X_eval.columns.get_loc(c) for c in cat_cols]
    }

    precomputed_stds = None
    if len(num_cols) > 0:
        precomputed_stds = X_eval.iloc[:, col_indices['num']].std().astype(np.float32)

    X_train_fit, y_train_fit = X_train, y_train
    if config.SVC_MAX_TRAIN_SAMPLES and len(X_train) > config.SVC_MAX_TRAIN_SAMPLES:
        strat = y_train if task_type == config.TASK_CLASSIFICATION else None
        X_train_fit, y_train_fit = resample(X_train, y_train, n_samples=config.SVC_MAX_TRAIN_SAMPLES,
                                            random_state=config.RANDOM_STATE, stratify=strat, replace=False)

    print(f"[{name}] Training base models ({task_type})...")
    pre = build_preprocessor(X_train_fit)
    models = build_models_fn(random_state=config.RANDOM_STATE)
    rows = []
    pipes = {}

    for model_name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        score, _ = eval_fn(pipe, X_train_fit, y_train_fit, X_test, y_test, refit=True)
        rows.append([name, task_type, config.CORRUPTION_CLEAN, 0.0, model_name, metric_name, score])
        pipes[model_name] = pipe

    temp_folder = tempfile.mkdtemp()
    try:
        X_eval_path = os.path.join(temp_folder, 'X_eval.mmap')
        y_eval_path = os.path.join(temp_folder, 'y_eval.mmap')

        dump(X_eval, X_eval_path)
        dump(y_eval, y_eval_path)

        X_eval_mmap = load(X_eval_path, mmap_mode='r')
        y_eval_mmap = load(y_eval_path, mmap_mode='r')

        tasks = []
        corruptions = CORRUPTION_FUNCS
        if task_type == config.TASK_REGRESSION:
            corruptions = {k: v for k, v in CORRUPTION_FUNCS.items() if
                           k not in (config.CORR_MISSING_MNAR, config.CORR_RARE_CLASS)}

        # Generate deterministic but unique seeds for every task.
        # This prevents correlation artifacts between different corruption types.
        # We generate a pool of seeds large enough for all potential tasks.
        master_rng = np.random.default_rng(config.RANDOM_STATE)
        # Estimate max tasks: 7 corruptions * 5 severities + mixed... 1000 is safe buffer
        seed_pool = master_rng.integers(0, 2**30, size=1000)
        seed_idx = 0

        for corr_name, corr_fn in corruptions.items():
            for s in severities:
                tasks.append((pipes, X_eval_mmap, y_eval_mmap, corr_name, s, corr_fn,
                              int(seed_pool[seed_idx]), metric_name,
                              task_type == config.TASK_CLASSIFICATION,
                              col_indices, None, precomputed_stds))
                seed_idx += 1

        if config.ENABLE_MIXED_SEVERITY:
            print("  -> generating mixed corruption tasks...")
            X_mixed, y_mixed = _get_eval_set(X_test, y_test, config.MIXED_SEVERITY_SAMPLE_LIMIT, config.RANDOM_STATE,
                                             stratify=stratify)

            X_mixed_path = os.path.join(temp_folder, 'X_mixed.mmap')
            y_mixed_path = os.path.join(temp_folder, 'y_mixed.mmap')
            dump(X_mixed, X_mixed_path)
            dump(y_mixed, y_mixed_path)

            X_mixed_mmap = load(X_mixed_path, mmap_mode='r')
            y_mixed_mmap = load(y_mixed_path, mmap_mode='r')

            corr_list = list(corruptions.items())
            for i in range(len(corr_list)):
                c1_name, c1_fn = corr_list[i]
                for j in range(i + 1, len(corr_list)):
                    c2_name, c2_fn = corr_list[j]
                    mid_s = severities[len(severities) // 2] if severities else 0.1

                    tasks.append((pipes, X_mixed_mmap, y_mixed_mmap, c1_name, mid_s, c1_fn,
                                  int(seed_pool[seed_idx]), metric_name,
                                  task_type == config.TASK_CLASSIFICATION,
                                  col_indices, (c2_name, c2_fn, mid_s), precomputed_stds))
                    seed_idx += 1

        print(f"[{name}] Running {len(tasks)} evaluation tasks (Parallel)...")

        batch_results = Parallel(n_jobs=config.BENCHMARK_N_JOBS, prefer="processes", batch_size=1)(
            delayed(_evaluate_group_task)(*t) for t in tasks
        )

        for group_res in batch_results:
            for r in group_res:
                rows.append([name] + r)

    finally:
        try:
            shutil.rmtree(temp_folder)
        except:
            pass
        gc.collect()

    df = pd.DataFrame(rows, columns=config.METRICS_COLUMNS)
    csv_name = config.METRICS_CSV_CLASSIFICATION if task_type == config.TASK_CLASSIFICATION else config.METRICS_CSV_REGRESSION
    df.to_csv(os.path.join(results_dir, csv_name.format(name=name)), index=False)
    plot_degradation(df, name, results_dir)
    return df


def run_classification_suite(name, X_train, X_test, y_train, y_test, results_dir, severities=config.DEFAULT_SEVERITIES,
                             random_state=config.RANDOM_STATE):
    return run_suite_generic(name, X_train, X_test, y_train, y_test, results_dir, severities,
                             config.TASK_CLASSIFICATION, build_classifiers, config.METRIC_AUC, evaluate_classifier)


def run_regression_suite(name, X_train, X_test, y_train, y_test, results_dir, severities=config.DEFAULT_SEVERITIES,
                         random_state=config.RANDOM_STATE):
    return run_suite_generic(name, X_train, X_test, y_train, y_test, results_dir, severities, config.TASK_REGRESSION,
                             build_regressors, config.METRIC_RMSE, evaluate_regressor)