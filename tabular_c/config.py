"""
Central configuration file for the TACO benchmark.
"""
import os
import sklearn
import torch
import pandas as pd

# --- Prevent Thread Oversubscription ---
# Forces single-threaded linear algebra within worker processes.
# This prevents the OS from thrashing when N_JOBS workers all try to use all cores.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Library-Level Thread Locking ---
# PyTorch and Pandas/NumPy can sometimes override env vars. We force them here.
torch.set_num_threads(1)

# Copy-on-Write (CoW) drastically reduces memory usage during slicing (Pandas >= 1.5)
try:
    pd.options.mode.copy_on_write = True
except (AttributeError, ValueError):
    pass

# --- Skip Scikit-Learn Input Checks ---
# Since we strictly handle missingness in our preprocessor, we disable
# finite checks (checking for NaNs/Infs) inside the models to save time.
sklearn.set_config(assume_finite=True)

# --- General Benchmark Config ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_SEVERITIES = (0.1, 0.2, 0.4)
QUICK_SEVERITIES = (0.2,)

# --- Performance / Speed Optimization Flags ---
ENABLE_MIXED_SEVERITY = True
HIGH_CARD_THRESHOLD = 100

# LIMITS
# Reduced slightly to prevent OOM on standard 16GB RAM machines during parallel execution
MIXED_SEVERITY_SAMPLE_LIMIT = 4000
EVAL_SAMPLE_LIMIT = 10000

TRY_INTEL_OPTIMIZATION = True

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

# Massive batch size for inference to saturate GPU
MODEL_FFN_BATCH = 16384

# --- Scalability Config ---
BENCHMARK_N_JOBS = -1
# Internal model jobs must be 1 because we parallelize at the corruption level.
MODEL_N_JOBS = 1

# SVM scales poorly O(n^2); limit training data specifically
SVC_MAX_TRAIN_SAMPLES = 20000
RF_ESTIMATORS = 100
XGB_ESTIMATORS = 200

# SAGA is faster for large datasets than liblinear
LOGREG_SOLVER = 'saga'
LOGREG_MAX_ITER = 5000

# --- Corruption Config ---
# Paper Section 4.2: Scaling factors for unit mismatch
SCALING_FACTORS = [0.45, 2.2, 3.28, 0.3048, 1.8, 0.5556]
DEFAULT_MINORITY_LABEL = 1

CORR_MISSING_MCAR = "missingness_mcar"
CORR_MISSING_MNAR = "missingness_mnar"
CORR_SCALING = "scaling_error"
CORR_CAT_REMAP = "categorical_remap"
CORR_NOISE_GAUSSIAN = "noise_gaussian"
CORR_NOISE_UNIFORM = "noise_uniform"
CORR_RARE_CLASS = "rare_class_dilution"

# --- Dataset Names ---
DATASET_FILENAME_MAP = {
    "adult": "adult.csv",
    "diabetes": "diabetes_130_us_hospitals.csv",
    "porto": "porto_seguro.csv",
    "ieee": "ieee_cis_fraud.csv",
    "nyc": "nyc_property_sales.csv",
}
DATASET_NAMES = list(DATASET_FILENAME_MAP.keys())