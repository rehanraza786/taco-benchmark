from . import config
from .utils import get_working_copy, get_smart_copy

import numpy as np
import pandas as pd

def apply_missingness(df: pd.DataFrame, y: pd.Series, severity=0.1, mnar=False, rng=None, inplace=False, col_idx=None, **kwargs):
    """
    Applies MCAR or MNAR missingness.
    Use flat numpy views and float32 generation to minimize RAM usage.
    """
    rng = np.random.default_rng(rng)
    out = get_working_copy(df, inplace)
    if severity <= 0: return out, y

    # MNAR: Conditional on Minority Class
    if mnar and y is not None:
        y_arr = y.values if hasattr(y, "values") else np.array(y)
        row_mask = (y_arr == config.DEFAULT_MINORITY_LABEL)

        if not np.any(row_mask): return out, y

        target_cols_idx = col_idx['num'] if col_idx else range(out.shape[1])

        if len(target_cols_idx) > 0:
            subset_vals = out.iloc[row_mask, target_cols_idx].values
            if subset_vals.dtype.kind == 'i':
                 subset_vals = subset_vals.astype(np.float32)

            n_cells = subset_vals.size
            n_mask = int(severity * n_cells)

            if n_mask > 0:
                flat_indices = rng.choice(n_cells, size=n_mask, replace=False)
                subset_vals.ravel()[flat_indices] = np.nan
                out.iloc[row_mask, target_cols_idx] = subset_vals

    # MCAR: Random global mask
    else:
        # 1. Numerical Block
        nums = col_idx['num'] if col_idx else []
        if len(nums) > 0:
            vals = out.iloc[:, nums].values
            if vals.dtype != np.float32:
                vals = vals.astype(np.float32)

            mask = rng.random(vals.shape, dtype=np.float32) < severity
            vals[mask] = np.nan
            out.iloc[:, nums] = vals

        # 2. Categorical Block
        cats = col_idx['cat'] if col_idx else []
        if len(cats) > 0:
            n_rows = len(out)
            for c_idx in cats:
                series = out.iloc[:, c_idx]
                if hasattr(series, "cat"):
                    codes = series.values.codes.copy()
                    mask = rng.random(n_rows, dtype=np.float32) < severity
                    codes[mask] = -1
                    out.iloc[:, c_idx] = pd.Categorical.from_codes(codes, categories=series.cat.categories)

    return out, y

def apply_scaling_errors(df: pd.DataFrame, y: pd.Series, severity=0.1, rng=None, inplace=False, col_idx=None, **kwargs):
    rng = np.random.default_rng(rng)
    out = get_smart_copy(df, inplace, include_dtypes=[np.number])

    nums = col_idx['num'] if col_idx else []
    n_nums = len(nums)
    if n_nums == 0: return out, y

    k = min(max(1, int(np.ceil(severity * n_nums))), n_nums)

    chosen_indices = rng.choice(nums, size=k, replace=False)
    factors = rng.choice(config.SCALING_FACTORS, size=k, replace=True)

    for idx, factor in zip(chosen_indices, factors):
        vals = out.iloc[:, idx].values
        if vals.dtype.kind != 'f':
            vals = vals.astype(np.float32)
        vals *= factor
        out.iloc[:, idx] = vals

    return out, y

def apply_categorical_remap(df: pd.DataFrame, y: pd.Series, severity=0.1, rng=None, inplace=False, col_idx=None, **kwargs):
    rng = np.random.default_rng(rng)
    out = get_smart_copy(df, inplace, include_dtypes=['category'])

    cats = col_idx['cat'] if col_idx else []
    if len(cats) == 0: return out, y

    k = min(max(1, int(np.ceil(severity * len(cats)))), len(cats))
    chosen_indices = rng.choice(cats, size=k, replace=False)

    for c_idx in chosen_indices:
        series = out.iloc[:, c_idx]
        if not hasattr(series, "cat"): continue

        n_cats = len(series.cat.categories)
        if n_cats <= 1: continue

        codes = series.values.codes.copy()
        perm = rng.permutation(n_cats)

        mask = codes >= 0
        if np.any(mask):
            codes[mask] = perm[codes[mask]]

        out.iloc[:, c_idx] = pd.Categorical.from_codes(codes, categories=series.cat.categories)

    return out, y

def apply_noise_injection(df, y, severity=0.1, gaussian=True, rng=None, inplace=False, precomputed_stds=None, col_idx=None, **kwargs):
    """
    Adds Gaussian or Uniform noise.
    OPTIMIZED: Strictly uses in-place operations and float32 buffers.
    """
    rng = np.random.default_rng(rng)
    out = get_smart_copy(df, inplace, include_dtypes=[np.number])

    nums = col_idx['num'] if col_idx else []
    if len(nums) == 0: return out, y

    vals = out.iloc[:, nums].values
    if vals.dtype != np.float32:
        vals = vals.astype(np.float32)

    if precomputed_stds is not None:
        if hasattr(precomputed_stds, "values"):
            stds = precomputed_stds.values.astype(np.float32)
        else:
            stds = np.array(precomputed_stds, dtype=np.float32)
    else:
        stds = np.nanstd(vals, axis=0).astype(np.float32)

    stds[stds == 0] = 1.0
    scaled_severity = (stds * severity).astype(np.float32)

    noise_buffer = np.empty_like(vals, dtype=np.float32)

    if gaussian:
        rng.standard_normal(out=noise_buffer, dtype=np.float32)
    else:
        rng.random(out=noise_buffer, dtype=np.float32)
        noise_buffer *= 2.0
        noise_buffer -= 1.0

    noise_buffer *= scaled_severity
    vals += noise_buffer
    out.iloc[:, nums] = vals

    return out, y

def apply_rare_class_dilution(X_test, y_test, severity=0.1, minority_label=config.DEFAULT_MINORITY_LABEL, rng=None, inplace=False, **kwargs):
    rng = np.random.default_rng(rng)

    if not inplace:
        X_test = X_test.copy(deep=True)
        y_test = y_test.copy(deep=True)

    if not isinstance(X_test, pd.DataFrame): X_test = pd.DataFrame(X_test)
    if not isinstance(y_test, pd.Series): y_test = pd.Series(y_test, index=X_test.index)

    mask_min = (y_test == minority_label)
    count_min = mask_min.sum()

    if count_min == 0: return X_test, y_test

    k = int(np.floor(severity * count_min))
    if k <= 0: return X_test, y_test

    min_indices = y_test.index[mask_min]
    to_drop = rng.choice(min_indices, size=k, replace=False)

    X_test.drop(index=to_drop, inplace=True)
    y_test.drop(index=to_drop, inplace=True)

    return X_test, y_test

CORRUPTION_FUNCS = {
    config.CORR_MISSING_MCAR: lambda X, y, s, rng=None, inplace=False, col_idx=None, **kwargs: apply_missingness(X, y, s, mnar=False, rng=rng, inplace=inplace, col_idx=col_idx, **kwargs),
    config.CORR_MISSING_MNAR: lambda X, y, s, rng=None, inplace=False, col_idx=None, **kwargs: apply_missingness(X, y, s, mnar=True, rng=rng, inplace=inplace, col_idx=col_idx, **kwargs),
    config.CORR_SCALING: lambda X, y, s, rng=None, inplace=False, col_idx=None, **kwargs: apply_scaling_errors(X, y, s, rng=rng, inplace=inplace, col_idx=col_idx, **kwargs),
    config.CORR_CAT_REMAP: lambda X, y, s, rng=None, inplace=False, col_idx=None, **kwargs: apply_categorical_remap(X, y, s, rng=rng, inplace=inplace, col_idx=col_idx, **kwargs),
    config.CORR_NOISE_GAUSSIAN: lambda X, y, s, rng=None, inplace=False, col_idx=None, **kwargs: apply_noise_injection(X, y, s, gaussian=True, rng=rng, inplace=inplace, col_idx=col_idx, **kwargs),
    config.CORR_NOISE_UNIFORM: lambda X, y, s, rng=None, inplace=False, col_idx=None, **kwargs: apply_noise_injection(X, y, s, gaussian=False, rng=rng, inplace=inplace, col_idx=col_idx, **kwargs),
    config.CORR_RARE_CLASS: lambda X, y, s, rng=None, inplace=False, col_idx=None, **kwargs: apply_rare_class_dilution(X, y, s, rng=rng, inplace=inplace, col_idx=col_idx, **kwargs),
}