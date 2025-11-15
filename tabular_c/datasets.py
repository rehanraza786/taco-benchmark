import os
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config  # <-- Import config

def infer_types(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")
    return df

def load_adult(path=None, random_state=config.RANDOM_STATE):
    if not os.path.exists(path):
        raise FileNotFoundError("Provide adult.csv")
    df = pd.read_csv(path)
    target = "income" if "income" in df.columns else ("class" if "class" in df.columns else None)
    if target is None:
        raise ValueError("adult.csv must have 'income' or 'class' target column.")
    y = (df[target].astype(str).str.contains(">50")).astype(int)
    X = df.drop(columns=[target])
    X = infer_types(X)
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)


def load_diabetes_130(path, random_state=config.RANDOM_STATE):
    if not os.path.exists(path):
        raise FileNotFoundError("Provide diabetes_130_us_hospitals.csv")
    df = pd.read_csv(path)
    if "readmitted" in df.columns:
        y = (df["readmitted"].astype(str).str.contains("<30")).astype(int)
        X = df.drop(columns=["readmitted"])
    elif "readmitted_flag" in df.columns:
        y = df["readmitted_flag"].astype(int)
        X = df.drop(columns=["readmitted_flag"])
    else:
        raise ValueError("Could not find target column ('readmitted' or 'readmitted_flag').")
    X = infer_types(X)
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)

def load_porto_seguro(path, random_state=config.RANDOM_STATE):
    if not os.path.exists(path):
        raise FileNotFoundError("Provide porto_seguro.csv (Kaggle)")
    df = pd.read_csv(path)
    target = "target"
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    X = infer_types(X)
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)

def load_ieee_cis(path, random_state=config.RANDOM_STATE):
    if not os.path.exists(path):
        raise FileNotFoundError("Provide ieee_cis_fraud.csv (Kaggle merged)")
    df = pd.read_csv(path)
    target = "isFraud"
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    X = infer_types(X)
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)

def load_nyc_property_sales(path, random_state=config.RANDOM_STATE):
    if not os.path.exists(path):
        raise FileNotFoundError("Provide nyc_property_sales.csv")
    df = pd.read_csv(path)
    possible_targets = ["SALE PRICE", "sale_price", "SALE_PRICE"]
    target = next((t for t in possible_targets if t in df.columns), None)
    if target is None:
        raise ValueError("Could not find sale price column in nyc_property_sales.csv")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    X = infer_types(X)
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state)