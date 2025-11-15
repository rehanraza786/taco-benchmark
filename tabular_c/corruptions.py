import numpy as np
import pandas as pd
from . import config
from .utils import deep_copy

def apply_missingness(df: pd.DataFrame, y: pd.Series, severity=0.1, mnar=False, rng=None):
    rng = np.random.default_rng(rng)
    out = deep_copy(df)
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if mnar and y is not None and len(np.unique(y))==2:
        idx_pos = df.index[y == 1]
        if len(idx_pos)==0:
            return out, y
        n = int(severity * df.loc[idx_pos].size)
        if n <= 0:
            return out, y
        flat_idx = rng.choice(np.arange(df.loc[idx_pos].size), size=n, replace=False)
        rows = flat_idx // df.shape[1]
        cols = flat_idx % df.shape[1]
        chosen_rows = idx_pos[rows]
        for r, c in zip(chosen_rows, cols):
            mask.iloc[df.index.get_loc(r), c] = True
    else:
        n = int(severity * df.size)
        if n <= 0:
            return out, y
        flat_idx = rng.choice(np.arange(df.size), size=n, replace=False)
        rows = flat_idx // df.shape[1]
        cols = flat_idx % df.shape[1]
        for r, c in zip(rows, cols):
            mask.iloc[r, c] = True
    out[mask] = np.nan
    return out, y

def apply_scaling_errors(df: pd.DataFrame, y: pd.Series, severity=0.1, rng=None):
    rng = np.random.default_rng(rng)
    out = deep_copy(df)
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return out, y
    k = max(1, int(np.ceil(severity * len(num_cols))))
    k = min(k, len(num_cols))
    chosen = rng.choice(num_cols, size=k, replace=False)
    factors = rng.choice(config.SCALING_FACTORS, size=k, replace=True)
    for col, f in zip(chosen, factors):
        out[col] = out[col] * f
    return out, y

def apply_categorical_remap(df: pd.DataFrame, y: pd.Series, severity=0.1, rng=None):
    rng = np.random.default_rng(rng)
    out = deep_copy(df)
    cat_cols = out.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        return out, y
    k = max(1, int(np.ceil(severity * len(cat_cols))))
    k = min(k, len(cat_cols))
    chosen = rng.choice(cat_cols, size=k, replace=False)
    for col in chosen:
        vals = out[col].dropna().unique().tolist()
        if len(vals) <= 1:
            continue
        perm = vals[:]
        rng.shuffle(perm)
        mapping = {v:p for v,p in zip(vals, perm)}
        out[col] = out[col].map(lambda x: mapping.get(x, x))
    return out, y

def apply_rare_class_dilution(X_test: pd.DataFrame, y_test: pd.Series, severity=0.1,
                              minority_label=config.DEFAULT_MINORITY_LABEL, rng=None):
    rng = np.random.default_rng(rng)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)

    idx_min = y_test[y_test == minority_label].index
    if len(idx_min) == 0:
        return X_test, y_test
    k = int(np.floor(severity * len(idx_min)))
    if k <= 0:
        return X_test, y_test
    to_drop = rng.choice(idx_min, size=k, replace=False)
    return X_test.drop(index=to_drop), y_test.drop(index=to_drop)

def apply_noise_injection(df: pd.DataFrame, y: pd.Series, severity=0.1, gaussian=True, rng=None):
    rng = np.random.default_rng(rng)
    out = deep_copy(df)
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        col_std = np.nanstd(out[col].values)
        if not np.isfinite(col_std) or col_std == 0:
            continue
        noise_std = severity * col_std
        if gaussian:
            noise = rng.normal(0.0, noise_std, size=len(out))
        else:
            noise = rng.uniform(-noise_std, noise_std, size=len(out))
        out[col] = out[col].astype(float) + noise
    return out, y

CORRUPTION_FUNCS = {
    config.CORR_MISSING_MCAR: lambda X, y, s, rng=None: apply_missingness(X, y, s, mnar=False, rng=rng),
    config.CORR_MISSING_MNAR: lambda X, y, s, rng=None: apply_missingness(X, y, s, mnar=True, rng=rng),
    config.CORR_SCALING: lambda X, y, s, rng=None: apply_scaling_errors(X, y, s, rng=rng),
    config.CORR_CAT_REMAP: lambda X, y, s, rng=None: apply_categorical_remap(X, y, s, rng=rng),
    config.CORR_NOISE_GAUSSIAN: lambda X, y, s, rng=None: apply_noise_injection(X, y, s, gaussian=True, rng=rng),
    config.CORR_NOISE_UNIFORM: lambda X, y, s, rng=None: apply_noise_injection(X, y, s, gaussian=False, rng=rng),
    config.CORR_RARE_CLASS: lambda X, y, s, rng=None: apply_rare_class_dilution(X, y, s, rng=rng),
}