import os
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def deep_copy(df: pd.DataFrame):
    return df.copy(deep=True)