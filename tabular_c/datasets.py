"""
Dataset loading utilities for TACO.
Includes optimized loaders for Adult, Diabetes, Porto Seguro, IEEE-CIS, and NYC Property.
"""
from . import config

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import load_cached

def load_adult(path=None, random_state=config.RANDOM_STATE):
    """Loads Adult Census dataset (Classification)."""
    df = load_cached(path)
    if df is None:
        raise FileNotFoundError("Provide adult.csv")

    target = "income" if "income" in df.columns else ("class" if "class" in df.columns else None)
    if target is None:
        raise ValueError("adult.csv must have 'income' or 'class' target column.")

    y = (df[target].astype(str).str.contains(">50")).astype(np.int8)
    X = df.drop(columns=[target])
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)


def load_diabetes_130(path, random_state=config.RANDOM_STATE):
    """Loads Diabetes 130-US Hospitals dataset (Classification)."""
    df = load_cached(path)
    if df is None:
        raise FileNotFoundError("Provide diabetes_130_us_hospitals.csv")

    if "readmitted" in df.columns:
        y = (df["readmitted"].astype(str).str.contains("<30")).astype(np.int8)
        X = df.drop(columns=["readmitted"])
    elif "readmitted_flag" in df.columns:
        y = df["readmitted_flag"].astype(np.int8)
        X = df.drop(columns=["readmitted_flag"])
    else:
        raise ValueError("Could not find target column ('readmitted' or 'readmitted_flag').")

    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)

def load_porto_seguro(path, random_state=config.RANDOM_STATE):
    """Loads Porto Seguro Safe Driver dataset (Classification)."""
    df = load_cached(path)
    if df is None:
        raise FileNotFoundError("Provide porto_seguro.csv")

    target = "target"
    y = df[target].astype(np.int8)
    X = df.drop(columns=[target])
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)

def load_ieee_cis(path, random_state=config.RANDOM_STATE):
    """Loads IEEE-CIS Fraud Detection dataset (Classification)."""
    df = load_cached(path)
    if df is None:
        raise FileNotFoundError("Provide ieee_cis_fraud.csv")

    target = "isFraud"
    y = df[target].astype(np.int8)
    X = df.drop(columns=[target])
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state, stratify=y)

def load_nyc_property_sales(path, random_state=config.RANDOM_STATE):
    """Loads NYC Property Sales dataset (Regression)."""
    df = load_cached(path)
    if df is None:
        raise FileNotFoundError("Provide nyc_property_sales.csv")

    possible_targets = ["SALE PRICE", "sale_price", "SALE_PRICE"]
    target = next((t for t in possible_targets if t in df.columns), None)
    if target is None:
        raise ValueError("Could not find sale price column in nyc_property_sales.csv")

    # Zero-copy filtering
    # 1. Coerce numeric in-place
    df[target] = pd.to_numeric(df[target], errors='coerce')

    # 2. Drop NaNs in-place (avoids creating a mask array and a new DataFrame copy)
    df.dropna(subset=[target], inplace=True)

    # 3. Pop target (In-place column removal, returns Series)
    y = df.pop(target).astype('float32')

    # 4. Remaining df is X (No extra copy created)
    return train_test_split(df, y, test_size=config.TEST_SIZE, random_state=random_state)