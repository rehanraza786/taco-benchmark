import os
import matplotlib.pyplot as plt
from .utils import ensure_dir


def plot_bar_clean(df, dataset_name, results_dir):
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
    plt.savefig(fp)
    plt.close()


def plot_degradation(df, dataset_name, results_dir):
    metric = df["metric"].iloc[0]
    models = sorted(df["model"].unique())
    corrs = sorted([c for c in df["corruption"].unique() if c != "clean"])

    plot_dir = os.path.join(results_dir, f"{dataset_name}_plots")
    ensure_dir(plot_dir)

    for corr in corrs:
        plt.figure(figsize=(10, 6))
        sub = df[df["corruption"] == corr]
        if sub.empty:
            continue

        # Try to convert to float for sorting, but keep original values
        try:
            # For single corruptions (0.1, 0.2, 0.4)
            svals_num = sub["severity"].unique().astype(float)
            svals = sorted(svals_num)
            is_categorical = False
        except ValueError:
            # For multi-corruptions ("0.1+0.1", "0.1+0.2", etc.)
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
        plt.savefig(out)
        plt.close()