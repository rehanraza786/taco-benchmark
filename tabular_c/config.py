"""
Central configuration file for the TACO benchmark.
Contains all magic strings, default parameters, and dataset definitions.
"""
# NOTE: This file should not import any other files from the tabular_c package
# to avoid circular imports.

# --- General Benchmark Config ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_SEVERITIES = (0.1, 0.2, 0.4)
QUICK_SEVERITIES = (0.2,)

# --- Directory Config ---
DEFAULT_DATA_DIR = "data"
DEFAULT_RESULTS_DIR = "results"

# --- Task & Metric Names ---
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
METRIC_AUC = "AUC"
METRIC_RMSE = "RMSE"
CORRUPTION_CLEAN = "clean"

# --- CSV Output Config ---
METRICS_COLUMNS = ["dataset", "task", "corruption", "severity", "model", "metric", "value"]
METRICS_CSV_CLASSIFICATION = "metrics_{name}_classification.csv"
METRICS_CSV_REGRESSION = "metrics_{name}_regression.csv"

# --- Model Config ---
DEFAULT_IMPUTE_STRATEGY = "median"
MODEL_LOGREG = "logreg"
MODEL_SVM = "svm"
MODEL_RF = "rf"
MODEL_XGB = "xgb"
MODEL_FFN = "ffn"
MODEL_LINREG = "linreg"
MODEL_SVR = "svr"
MODEL_RF_REG = "rf_reg"
MODEL_XGB_REG = "xgb_reg"
MODEL_FFN_REG = "ffn_reg"

# --- Corruption Config ---
SCALING_FACTORS = [0.45, 2.2, 3.28, 0.3048, 1.8, 0.5556]
DEFAULT_MINORITY_LABEL = 1

CORR_MISSING_MCAR = "missingness_mcar"
CORR_MISSING_MNAR = "missingness_mnar"
CORR_SCALING = "scaling_error"
CORR_CAT_REMAP = "categorical_remap"
CORR_NOISE_GAUSSIAN = "noise_gaussian"
CORR_NOISE_UNIFORM = "noise_uniform"
CORR_RARE_CLASS = "rare_class_dilution"

# --- Dataset Names (for file paths) ---
DATASET_FILENAME_MAP = {
    "adult": "adult.csv",
    "diabetes": "diabetes_130_us_hospitals.csv",
    "porto": "porto_seguro.csv",
    "ieee": "ieee_cis_fraud.csv",
    "nyc": "nyc_property_sales.csv",
}
DATASET_NAMES = list(DATASET_FILENAME_MAP.keys())