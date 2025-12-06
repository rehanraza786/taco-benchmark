import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px

st.set_page_config(layout="wide", page_title="TACO Benchmark Results")
st.title("🌮 TACO: Tabular Corruptions Benchmark")

@st.cache_data
def load_results(results_dir="results"):
    files = glob.glob(os.path.join(results_dir, "metrics_*.csv"))
    if not files:
        return pd.DataFrame()

    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f))
    return pd.concat(df_list, ignore_index=True)


df = load_results()

if df.empty:
    st.error("No results found in 'results/'. Run the benchmark first!")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters")
dataset = st.sidebar.selectbox("Dataset", df["dataset"].unique())
task_df = df[df["dataset"] == dataset]

corruptions = [c for c in task_df["corruption"].unique() if c != "clean"]
selected_corr = st.sidebar.selectbox("Corruption Type", corruptions)

metric = task_df["metric"].iloc[0]

clean_baseline = task_df[task_df["corruption"] == "clean"]
corrupted_data = task_df[task_df["corruption"] == selected_corr].copy()

# Determine if numeric or categorical severity
try:
    # Try converting to float for sorting
    corrupted_data["severity_num"] = corrupted_data["severity"].astype(float)
    x_axis = "severity_num"
    is_cat = False
except ValueError:
    # Handle "0.1+0.1" mixed severities
    x_axis = "severity"
    is_cat = True

st.subheader(f"Degradation: {dataset} - {selected_corr}")

fig = px.line(
    corrupted_data.sort_values(by=x_axis),
    x=x_axis,
    y="value",
    color="model",
    markers=True,
    title=f"Performance vs Severity ({metric})",
    labels={"value": metric, x_axis: "Severity"}
)

if st.checkbox("Show Clean Baselines"):
    for model in clean_baseline["model"].unique():
        val = clean_baseline[clean_baseline["model"] == model]["value"].values[0]
        fig.add_hline(y=val, line_dash="dot", annotation_text=f"{model} (Clean)", annotation_position="top right")

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Raw Data")
sorted_df = corrupted_data.sort_values(["model", x_axis])
st.dataframe(sorted_df[["model", "severity", "value", "metric"]])