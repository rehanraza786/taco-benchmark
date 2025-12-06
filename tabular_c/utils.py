import os
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_working_copy(df, inplace):
    return df if inplace else df.copy(deep=True)


def get_smart_copy(df, inplace, include_dtypes=None):
    if inplace: return df
    out = df.copy(deep=False)
    if include_dtypes:
        cols = df.select_dtypes(include=include_dtypes).columns
        if len(cols) > 0:
            out[cols] = df[cols].copy(deep=True)
    return out


def infer_types(df: pd.DataFrame):
    """
    Optimized type inference.
    Vectorized object->category conversion.
    NumPy-based min/max check for integer downcasting (skips Pandas overhead).
    """
    # Object -> Category (Vectorized)
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].astype("category")

    # Float64 -> Float32
    fcols = df.select_dtypes(include=['float64']).columns
    if len(fcols) > 0:
        df[fcols] = df[fcols].astype('float32')

    # Int64 -> Downcast (Fast Manual Checks via NumPy)
    icols = df.select_dtypes(include=['int64', 'int']).columns
    for col in icols:
        # Access .values to skip Pandas Series overhead
        vals = df[col].values
        c_min, c_max = vals.min(), vals.max()

        if c_min >= -128 and c_max <= 127:
            df[col] = df[col].astype('int8')
        elif c_min >= -32768 and c_max <= 32767:
            df[col] = df[col].astype('int16')
        elif c_min >= -2147483648 and c_max <= 2147483647:
            df[col] = df[col].astype('int32')

    return df


def load_cached(path):
    if not os.path.exists(path): return None
    pq_path = path.replace(".csv", ".parquet")

    # Try Parquet (Fastest)
    if os.path.exists(pq_path):
        try:
            return pd.read_parquet(pq_path, engine='pyarrow')
        except Exception:
            pass

    try:
        # Pyarrow engine is much faster for CSVs
        df = pd.read_csv(path, engine='pyarrow')
    except (ValueError, ImportError):
        # Fallback to standard C engine
        df = pd.read_csv(path)

    df = infer_types(df)

    try:
        df.to_parquet(pq_path, index=False, engine='pyarrow', compression='zstd')
    except Exception:
        pass

    return df