# TACO Benchmark Specification

This document describes the core expectations for the TACO (Tabular Corruptions)
benchmark so that others can reproduce your setup or extend it.

## Goals

- Provide an ImageNet-C–style robustness benchmark for tabular models.
- Standardize:
  - Datasets and splits
  - Corruptions and severity levels
  - Model families and default hyperparameters
  - Evaluation metrics and file formats

## Datasets

Current reference datasets:

- **Adult** (UCI) — binary classification, income > 50K
- **Diabetes 130-US Hospitals** — readmission classification
- **Porto Seguro Safe Driver** — insurance claim classification
- **IEEE-CIS Fraud Detection** — transaction fraud classification
- **NYC Property Sales** — log sale price regression

All datasets are expected as CSVs in `./data/`.

## Corruptions

Implemented in `tabular_c/corruptions.py` and exposed via `CORRUPTION_FUNCS`.

Each corruption is a function:

```python
def fn(X: pd.DataFrame, y, severity: float, rng=None) -> tuple[pd.DataFrame, pd.Series]:
    ...
```

with `severity` in `{0.1, 0.2, 0.4}` by default.

Corruption families include (non-exhaustive):

- Missingness (MCAR, MNAR)
- Scaling errors (unit mismatches)
- Categorical remapping
- Noise injection (Gaussian, uniform)
- Rare-class dilution (classification only)

## Evaluation protocol

- Train models on **clean** training data only.
- Evaluate on:
  - Clean test data (baseline)
  - Corrupted test data for each corruption × severity
  - Optionally, multi-corruption compositions

For each dataset and model:

- For classification: record AUC, accuracy, precision, recall and confusion matrices.
- For regression: record RMSE.

Metrics are saved as CSVs (one row per `(dataset, model, corruption, severity, metric)`)
in a `results/` directory (created if needed).

## Models

At minimum, the following model families are supported:

- Logistic Regression / Linear Regression
- SVM / SVR (RBF)
- Random Forest (Classifier / Regressor)
- XGBoost (Classifier / Regressor)
- Feedforward neural network (PyTorch)

## Reproducibility

- Use a fixed random seed (e.g. `random_state=42`) for splits and corruptions.
- Do not leak corrupted data into training; all corruptions apply to test only.
- Keep the CSV layout stable so downstream analysis scripts can rely on it.
