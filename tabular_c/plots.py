"""
Plotting utilities for visualization of benchmark results.
"""
import os
import matplotlib

# Use non-interactive backend.
# This is 2-3x faster for saving files and prevents crashes on headless servers.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from .utils import ensure_dir

def plot_bar_clean(df, dataset_name, results_dir):
    """Plots a bar chart of clean baseline performance across models."""
    met = df["metric"].iloc[0]
    clean = df[df["corruption"] == "clean"]

    plt.figure()
    xs = range(len(clean))
    plt.bar(xs, clean["value"])
    plt.xticks(xs, clean["model"], rotation=45, ha="right")
    plt.ylabel(met)
    plt.title(f"{dataset_name} — Clean {met} by Model")
    plt.tight_layout()

    fp = os.path.join(results_dir, f"{dataset_name}_clean_{met}_bar.png")
    plt.savefig(fp, dpi=150)
    plt.close()


def plot_degradation(df, dataset_name, results_dir):
    """
    Plots performance degradation curves for each corruption type.
    """
    metric = df["metric"].iloc[0]
    models = sorted(df["model"].unique())
    corrs = sorted([c for c in df["corruption"].unique() if c != "clean"])

    plot_dir = os.path.join(results_dir, f"{dataset_name}_plots")
    ensure_dir(plot_dir)

    for corr in corrs:
        fig = plt.figure(figsize=(10, 6))
        sub = df[df["corruption"] == corr]
        if sub.empty:
            plt.close(fig)
            continue

        try:
            svals_num = sub["severity"].unique().astype(float)
            svals = sorted(svals_num)
            is_categorical = False
        except ValueError:
            svals = sorted(sub["severity"].unique(), key=lambda s: tuple(map(float, s.split('+'))))
            is_categorical = True

        for m in models:
            model_data = sub[sub['model'] == m].groupby("severity")['value'].mean()
            yvals = model_data.reindex(svals).values

            if is_categorical:
                plt.plot(svals, yvals, marker='o', label=m, linestyle='None')
            else:
                plt.plot(svals, yvals, marker='o', label=m)

        corr_filename = corr.replace("+", "_plus_")
        plt.title(f"{dataset_name} — {corr} ({metric} vs severity)")

        if is_categorical:
            plt.xlabel("Severity Pair (s1+s2)")
            plt.xticks(rotation=45, ha="right")
        else:
            plt.xlabel("Severity")

        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(plot_dir, f"{dataset_name}_{corr_filename}_{metric}_curve.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)