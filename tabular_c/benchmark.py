import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from . import config
from .corruptions import CORRUPTION_FUNCS
from .models import build_preprocessor, build_classifiers, build_regressors, evaluate_classifier, evaluate_regressor
from .utils import ensure_dir
from .plots import plot_degradation


def run_classification_suite(name, X_train, X_test, y_train, y_test, results_dir,
                             severities=config.DEFAULT_SEVERITIES,
                             random_state=config.RANDOM_STATE):
    ensure_dir(results_dir)
    pre = build_preprocessor(X_train)
    clfs = build_classifiers(random_state=random_state)

    rows = []
    for model_name, clf in clfs.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        # Clean baseline
        print(f"[{name}] Running {model_name} (clean)...")
        auc_clean, proba_clean = evaluate_classifier(pipe, X_train, y_train, X_test, y_test, refit=True)
        rows.append(
            [name, config.TASK_CLASSIFICATION, config.CORRUPTION_CLEAN, 0.0, model_name, config.METRIC_AUC, auc_clean])

        # Save confusion matrix
        y_pred_clean = (proba_clean >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_clean)
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"{name}-{model_name}-clean Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{name}_{model_name}_clean_confusion.png"))
        plt.close()

        # --- Single Corruptions ---
        print(f"[{name}] Running {model_name} (single corruptions)...")
        for corr_name, corr_fn in CORRUPTION_FUNCS.items():
            for s in severities:
                Xc, yc = corr_fn(X_test, y_test, s, rng=random_state)
                auc_corr, _ = evaluate_classifier(pipe, X_train, y_train, Xc, yc, refit=False)
                rows.append([name, config.TASK_CLASSIFICATION, corr_name, s, model_name, config.METRIC_AUC, auc_corr])

        # --- Linear-Severity Multi-Corruption ---
        print(f"[{name}] Running {model_name} (linear-severity multi-corruptions)...")
        for s in severities:
            for (c1_name, c1_fn), (c2_name, c2_fn) in itertools.combinations(CORRUPTION_FUNCS.items(), 2):
                Xc1, yc1 = c1_fn(X_test, y_test, s, rng=random_state)
                Xc2, yc2 = c2_fn(Xc1, yc1, s, rng=random_state)
                auc_multi, _ = evaluate_classifier(pipe, X_train, y_train, Xc2, yc2, refit=False)
                combo_name = f"{c1_name}+{c2_name}"
                s_label = f"{s:.1f}+{s:.1f}"
                rows.append(
                    [name, config.TASK_CLASSIFICATION, combo_name, s_label, model_name, config.METRIC_AUC, auc_multi])

        # --- Mixed-Severity Multi-Corruption ---
        print(f"[{name}] Running {model_name} (mixed-severity multi-corruptions)...")
        for (c1_name, c1_fn), (c2_name, c2_fn) in itertools.combinations(CORRUPTION_FUNCS.items(), 2):
            for s1, s2 in itertools.product(severities, severities):
                if s1 == s2:
                    continue  # Already handled

                Xc1, yc1 = c1_fn(X_test, y_test, s1, rng=random_state)
                Xc2, yc2 = c2_fn(Xc1, yc1, s2, rng=random_state)
                auc_multi, _ = evaluate_classifier(pipe, X_train, y_train, Xc2, yc2, refit=False)
                combo_name = f"{c1_name}+{c2_name}"
                s_label = f"{s1:.1f}+{s2:.1f}"
                rows.append(
                    [name, config.TASK_CLASSIFICATION, combo_name, s_label, model_name, config.METRIC_AUC, auc_multi])

    df = pd.DataFrame(rows, columns=config.METRICS_COLUMNS)
    out_csv = os.path.join(results_dir, config.METRICS_CSV_CLASSIFICATION.format(name=name))
    df.to_csv(out_csv, index=False)
    print(f"[{name}] classification metrics saved -> {out_csv}")
    plot_degradation(df, name, results_dir)
    return df


def run_regression_suite(name, X_train, X_test, y_train, y_test, results_dir,
                         severities=config.DEFAULT_SEVERITIES,
                         random_state=config.RANDOM_STATE):
    ensure_dir(results_dir)
    pre = build_preprocessor(X_train)
    regs = build_regressors(random_state=random_state)

    reg_corruptions = {
        k: v for k, v in CORRUPTION_FUNCS.items()
        if k not in (config.CORR_MISSING_MNAR, config.CORR_RARE_CLASS)
    }

    rows = []
    for model_name, reg in regs.items():
        pipe = Pipeline([("pre", pre), ("reg", reg)])

        # Clean baseline
        print(f"[{name}] Running {model_name} (clean)...")
        rmse_clean, _ = evaluate_regressor(pipe, X_train, y_train, X_test, y_test, refit=True)
        rows.append(
            [name, config.TASK_REGRESSION, config.CORRUPTION_CLEAN, 0.0, model_name, config.METRIC_RMSE, rmse_clean])

        # --- Single Corruptions ---
        print(f"[{name}] Running {model_name} (single corruptions)...")
        for corr_name, corr_fn in reg_corruptions.items():
            for s in severities:
                Xc, _ = corr_fn(X_test, None, s, rng=random_state)
                # <-- FIXED: Pass refit=False
                rmse_corr, _ = evaluate_regressor(pipe, X_train, y_train, Xc, y_test, refit=False)
                rows.append([name, config.TASK_REGRESSION, corr_name, s, model_name, config.METRIC_RMSE, rmse_corr])

        # --- Linear-Severity Multi-Corruption ---
        print(f"[{name}] Running {model_name} (linear-severity multi-corruptions)...")
        for s in severities:
            for (c1_name, c1_fn), (c2_name, c2_fn) in itertools.combinations(reg_corruptions.items(), 2):
                Xc1, _ = c1_fn(X_test, None, s, rng=random_state)
                Xc2, _ = c2_fn(Xc1, None, s, rng=random_state)
                rmse_multi, _ = evaluate_regressor(pipe, X_train, y_train, Xc2, y_test, refit=False)
                combo_name = f"{c1_name}+{c2_name}"
                s_label = f"{s:.1f}+{s:.1f}"
                rows.append(
                    [name, config.TASK_REGRESSION, combo_name, s_label, model_name, config.METRIC_RMSE, rmse_multi])

        # --- Mixed-Severity Multi-Corruption ---
        print(f"[{name}] Running {model_name} (mixed-severity multi-corruptions)...")
        for (c1_name, c1_fn), (c2_name, c2_fn) in itertools.combinations(reg_corruptions.items(), 2):
            for s1, s2 in itertools.product(severities, severities):
                if s1 == s2:
                    continue

                Xc1, _ = c1_fn(X_test, None, s1, rng=random_state)
                Xc2, _ = c2_fn(Xc1, None, s2, rng=random_state)
                rmse_multi, _ = evaluate_regressor(pipe, X_train, y_train, Xc2, y_test, refit=False)
                combo_name = f"{c1_name}+{c2_name}"
                s_label = f"{s1:.1f}+{s2:.1f}"
                rows.append(
                    [name, config.TASK_REGRESSION, combo_name, s_label, model_name, config.METRIC_RMSE, rmse_multi])

    df = pd.DataFrame(rows, columns=config.METRICS_COLUMNS)
    out_csv = os.path.join(results_dir, config.METRICS_CSV_REGRESSION.format(name=name))
    df.to_csv(out_csv, index=False)
    print(f"[{name}] regression metrics saved -> {out_csv}")
    plot_degradation(df, name, results_dir)
    return df