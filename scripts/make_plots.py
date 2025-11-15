"""
Utility script to generate plots from TACO benchmark results.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tabular_c import config

def load_metrics(path: str | Path) -> pd.DataFrame:
    """Load a metrics CSV into a DataFrame."""
    path = Path(path)
    return pd.read_csv(path)

def plot_adult_robustness_curves(
    metrics_path: str | Path,
    output_path: str | Path = "figs/adult_robustness_curves.pdf",
    models=(config.MODEL_LOGREG, config.MODEL_RF, config.MODEL_XGB, config.MODEL_FFN),
    corruptions=(config.CORR_NOISE_GAUSSIAN, config.CORR_CAT_REMAP),
    metric_name=config.METRIC_AUC, # <-- Use constant
) -> None:
    """
    Plot AUC vs severity for several models on Adult under two corruptions.
    """
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_path)

    df = df[
        (df["dataset"] == "adult")
        & (df["task"] == config.TASK_CLASSIFICATION)
        & (df["metric"] == metric_name)
        & (df["corruption"].isin(list(corruptions) + [config.CORRUPTION_CLEAN]))
    ].copy()

    # Ensure severity is numeric
    df["severity"] = df["severity"].astype(float)

    fig, axes = plt.subplots(1, len(corruptions), figsize=(10, 4), sharey=True)
    if len(corruptions) == 1:
        axes = [axes]

    for ax, corr in zip(axes, corruptions):
        for model in models:
            df_model_clean = df[(df["model"] == model) & (df["corruption"] == config.CORRUPTION_CLEAN)]
            df_model_corr = df[
                (df["model"] == model) & (df["corruption"] == corr)
            ].sort_values("severity")

            if df_model_corr.empty:
                print(f"Warning: No data for model='{model}', corruption='{corr}'")
                continue

            x = df_model_corr["severity"].tolist()
            y = df_model_corr["value"].tolist()

            if not df_model_clean.empty:
                clean_val = df_model_clean["value"].iloc[0]
                if 0.0 not in x:
                    x = [0.0] + x
                    y = [clean_val] + y

            ax.plot(x, y, marker="o", label=model)

        ax.set_title(corr.replace("_", " ").title())
        ax.set_xlabel("Severity")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_xticks(config.DEFAULT_SEVERITIES)
        ax.set_xticks([0.0] + list(config.DEFAULT_SEVERITIES), minor=False)
        ax.set_xticklabels([0.0] + list(config.DEFAULT_SEVERITIES))


    axes[0].set_ylabel(metric_name.upper())

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(models), 4),
            bbox_to_anchor=(0.5, 1.05),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_robustness_scores(df: pd.DataFrame, metric_name: str = config.METRIC_AUC) -> pd.DataFrame:
    """
    Compute robustness scores R_{m,c}
    """
    df = df[df["metric"] == metric_name].copy()

    df_clean = df[df["corruption"] == config.CORRUPTION_CLEAN]
    clean_map = df_clean.groupby("model")["value"].mean().to_dict()

    df_corr = df[df["corruption"] != config.CORRUPTION_CLEAN].copy()
    df_corr = df_corr[df_corr["model"].isin(clean_map.keys())]

    def norm_row(row):
        # Add safety check for division by zero
        clean_val = clean_map.get(row["model"])
        if clean_val is None or clean_val == 0:
            return 0.0
        return row["value"] / clean_val

    df_corr["normalized"] = df_corr.apply(norm_row, axis=1)

    scores = (
        df_corr.groupby(["model", "corruption"])["normalized"]
        .mean()
        .reset_index()
        .rename(columns={"normalized": "robustness_score"})
    )
    return scores


def plot_adult_robustness_bars(
    metrics_path: str | Path,
    output_path: str | Path = "paper/figs/adult_robustness_bars.pdf",
    models=(config.MODEL_LOGREG, config.MODEL_RF, config.MODEL_XGB, config.MODEL_FFN),
    metric_name=config.METRIC_AUC,
) -> None:
    """
    Plot robustness scores R_{m,c} as a grouped bar chart for the Adult dataset.
    """
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_path)

    df = df[(df["dataset"] == "adult") & (df["task"] == config.TASK_CLASSIFICATION)].copy()
    scores = compute_robustness_scores(df, metric_name=metric_name)
    scores = scores[scores["model"].isin(models)]

    pivot = scores.pivot(index="model", columns="corruption", values="robustness_score").fillna(0.0)
    pivot = pivot.reindex(models) # Ensure consistent model order

    corruption_names = [c for c in pivot.columns if c != config.CORRUPTION_CLEAN]
    if not corruption_names:
        return
    pivot = pivot[corruption_names]

    num_models = pivot.shape[0]
    num_corrs = pivot.shape[1]

    fig, ax = plt.subplots(figsize=(10, 4))

    x_positions = list(range(num_models))
    width = 0.8 / max(num_corrs, 1)

    for i, corr in enumerate(corruption_names):
        offsets = [x + (i - (num_corrs - 1) / 2) * width for x in x_positions]
        ax.bar(
            offsets,
            pivot[corr].values,
            width=width,
            label=corr.replace("_", " ").title(),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Robustness score $R_{m,c}$")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Corruption", ncol=min(num_corrs, 4))

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    metrics_file = Path(config.DEFAULT_RESULTS_DIR) / config.METRICS_CSV_CLASSIFICATION.format(name="adult")
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Expected metrics file at {metrics_file}. "
            "Run the benchmark first (e.g., `python -m tabular_c.cli --only adult`)."
        )

    plot_adult_robustness_curves(metrics_file)
    plot_adult_robustness_bars(metrics_file)
    print("Saved figures to figs/")