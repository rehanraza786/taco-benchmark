import os
import sys
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Path Setup
# -----------------------------------------------------------------------------
# Get the directory where this script resides (root/scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (root/)
project_root = os.path.dirname(script_dir)

# Add project root to sys.path so we can import 'tabular_c' modules if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import config loader from your package
try:
    from tabular_c.utils import load_config
except ImportError:
    # Fallback if package isn't installed in environment
    import yaml


    def load_config(path):
        with open(path, 'r') as f: return yaml.safe_load(f)


def main():
    # -------------------------------------------------------------------------
    # 2. Configuration & Directory Resolution
    # -------------------------------------------------------------------------
    # User specified config is in 'tabular_c' directory
    config_path = os.path.join(project_root, 'tabular_c', 'config.yaml')

    # User specified results is in root directory
    results_dir = os.path.join(project_root, 'results')

    # Optional: If config exists, check if it overrides results_dir
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
        # If config has a results_dir setting, use it, otherwise stick to root/results
        cfg_dir = config.get('results_dir')
        if cfg_dir:
            # Handle relative path in config
            if os.path.isabs(cfg_dir):
                results_dir = cfg_dir
            else:
                results_dir = os.path.join(project_root, cfg_dir)

    print(f"Reading results from: {results_dir}")

    # -------------------------------------------------------------------------
    # 3. Dynamic Data Loading
    # -------------------------------------------------------------------------
    # Find all CSVs matching the pattern
    csv_pattern = os.path.join(results_dir, "metrics_*.csv")
    files = glob.glob(csv_pattern)

    if not files:
        print(f"No files found matching {csv_pattern}")
        return

    dataframes = []
    for filename in files:
        try:
            df = pd.read_csv(filename)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not dataframes:
        print("No valid dataframes loaded.")
        return

    full_df = pd.concat(dataframes, ignore_index=True)
    full_df['value'] = pd.to_numeric(full_df['value'], errors='coerce')

    # -------------------------------------------------------------------------
    # 4. Data Preparation (Subsets)
    # -------------------------------------------------------------------------
    # Baseline (Clean)
    clean_df = full_df[full_df['corruption'] == 'clean']

    # Stress Test (Mixed Corruptions)
    # Checks for explicit '0.2+0.2' severity or string containment
    stress_df = full_df[
        (full_df['severity'] == '0.2+0.2') |
        (full_df['severity'].astype(str).str.contains(r'\+', regex=True))
        ].copy()
    stress_agg = stress_df.groupby(['dataset', 'model'])['value'].mean().reset_index()

    # MCAR Robustness
    mcar_df = full_df[full_df['corruption'] == 'missingness_mcar'].copy()
    mcar_df['severity'] = pd.to_numeric(mcar_df['severity'], errors='coerce')

    # Add clean baseline (severity 0.0) to MCAR plot data
    clean_for_mcar = clean_df.copy()
    clean_for_mcar['corruption'] = 'missingness_mcar'
    clean_for_mcar['severity'] = 0.0
    mcar_plot_df = pd.concat([clean_for_mcar, mcar_df], ignore_index=True)

    # -------------------------------------------------------------------------
    # 5. Plotting Logic
    # -------------------------------------------------------------------------
    # Define datasets to plot
    datasets = ['porto', 'nyc', 'ieee', 'diabetes', 'adult']
    dataset_titles = {
        'porto': 'Porto',
        'nyc': 'NYC',
        'ieee': 'IEEE',
        'diabetes': 'Diabetes',
        'adult': 'Adult'
    }

    # Setup output directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    sns.set(style="whitegrid")

    # Figure Setup
    fig = plt.figure(figsize=(24, 11))
    gs = fig.add_gridspec(5, 3, wspace=0.25, hspace=0.5)

    for i, ds in enumerate(datasets):
        # --- Column 1: Baseline (Clean) ---
        ax1 = fig.add_subplot(gs[i, 0])
        subset_c = clean_df[clean_df['dataset'] == ds].sort_values('value', ascending=(ds == 'nyc'))

        if not subset_c.empty:
            # FIX: Added hue='model' and legend=False to fix deprecation warning
            sns.barplot(x='model', y='value', hue='model', data=subset_c, ax=ax1, palette='viridis', legend=False)

            ax1.set_ylabel('RMSE' if ds == 'nyc' else 'AUC', fontsize=12)
            ax1.set_xlabel('')
            # Add Row Title on the left
            ax1.text(-0.25, 0.5, dataset_titles[ds], transform=ax1.transAxes,
                     fontsize=16, fontweight='bold', va='center', rotation=90)
            if i == 0:
                ax1.set_title('Baseline (Clean)', fontsize=18, fontweight='bold')

        # --- Column 2: Robustness (MCAR) ---
        ax2 = fig.add_subplot(gs[i, 1])
        subset_m = mcar_plot_df[mcar_plot_df['dataset'] == ds]

        if not subset_m.empty:
            sns.lineplot(x='severity', y='value', hue='model', marker='o',
                         data=subset_m, ax=ax2, legend=(i == 0))
            ax2.set_ylabel('')
            ax2.set_xlabel('')
            if i == 0:
                ax2.set_title('Robustness (Missing)', fontsize=18, fontweight='bold')
                # Legend adjustment
                ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

        # --- Column 3: Stress Test (Mixed) ---
        ax3 = fig.add_subplot(gs[i, 2])
        subset_s = stress_agg[stress_agg['dataset'] == ds].sort_values('value', ascending=(ds == 'nyc'))

        if not subset_s.empty:
            # FIX: Added hue='model' and legend=False to fix deprecation warning
            sns.barplot(x='model', y='value', hue='model', data=subset_s, ax=ax3, palette='magma', legend=False)

            ax3.set_ylabel('')
            ax3.set_xlabel('')
            if i == 0:
                ax3.set_title('Stress Test (Mixed)', fontsize=18, fontweight='bold')

    # Save Output
    output_path = os.path.join(plots_dir, 'graphs.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plots saved successfully to: {output_path}")


if __name__ == "__main__":
    main()