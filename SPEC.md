# TACO Benchmark Specification

This document describes the core expectations for the TACO (Tabular Corruptions)
benchmark to ensure reproducibility and standardized evaluation.

## 🎯 Goals

- Provide an ImageNet-C–style robustness benchmark specifically for tabular data.
- Standardize:
  - **Data Loading:** Zero-copy, type-safe loading with stratification.
  - **Corruptions:** Deterministic application of realistic data faults.
  - **Evaluation:** Unified metrics (AUC/RMSE) across diverse model families.

## 💾 Datasets

| Dataset | Task | Metric | Source |
| :--- | :--- | :--- | :--- |
| **Adult** | Classification | AUC | UCI |
| **Diabetes 130-US** | Classification | AUC | UCI |
| **Porto Seguro** | Classification | AUC | Kaggle |
| **IEEE-CIS Fraud** | Classification | AUC | Kaggle |
| **NYC Property** | Regression | RMSE | NYC Open Data |

*Note: All datasets are expected as CSVs in `./data/`.*

## 🧪 Corruptions

Implemented in `tabular_c/corruptions.py`. Default severities are `{0.1, 0.2, 0.4}`.

1.  **Missingness (MCAR):** Randomly drops values across columns.
2.  **Missingness (MNAR):** Drops values conditional on the minority class (classification only).
3.  **Scaling Errors:** Multiplies numerical features by random unit conversion factors (e.g., meters to feet).
4.  **Categorical Remapping:** Shuffles the integer mapping of categorical encodings.
5.  **Noise Injection:** Adds Gaussian or Uniform noise scaled to the feature's standard deviation.
6.  **Rare Class Dilution:** Randomly removes instances of the minority class (Classification only).
7.  **Mixed Severity:** Applies two corruptions sequentially (e.g., Missingness + Noise).

## ⚙️ Models

The benchmark evaluates the following families (implemented in `tabular_c/models.py`):

- **Linear:** Logistic Regression / Linear Regression (SAGA solver).
- **SVM:** SGD-based approximation with Nystroem kernel approximation.
- **Trees:** Random Forest and XGBoost (Histogram-based).
- **Deep Learning:** Feedforward Neural Network (PyTorch) with large batch inference.

## 🔁 Reproducibility Protocol

1.  **Seeding:** A master `RANDOM_STATE` (default 42) generates a unique, deterministic integer seed for *every* specific corruption task. This ensures that "Missingness" and "Noise" do not share the same entropy source.
2.  **Splits:** Train/Test splits are stratified and fixed.
3.  **Isolation:** Corruptions are applied **only** to the test set at inference time. The training set remains clean.
4.  **Environment:** Threading is explicitly locked to `1` per worker (via `OMP_NUM_THREADS`, `torch.set_num_threads`) to prevent OS thrashing during parallel execution.