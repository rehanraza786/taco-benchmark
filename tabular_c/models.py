from . import config

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if config.TRY_INTEL_OPTIMIZATION:
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        print("  -> Intel(R) Extension for Scikit-learn enabled.")
    except ImportError:
        pass

def build_preprocessor(X: pd.DataFrame, numeric_strategy=config.DEFAULT_IMPUTE_STRATEGY):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if X[c].nunique() < config.HIGH_CARD_THRESHOLD]

    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        # OPTIMIZATION: copy=False allows in-place scaling if the memory layout permits.
        ("scaler", StandardScaler(copy=False))
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8))
    ])

    transformers = [("num", num_tf, num_cols)]
    if cat_cols:
        transformers.append(("cat", cat_tf, cat_cols))

    return ColumnTransformer(transformers=transformers, n_jobs=config.MODEL_N_JOBS, verbose_feature_names_out=False)


class TorchMLPBase(BaseEstimator):
    """
    Shared logic for Regressor and Classifier.
    Uses torch.inference_mode(), reduced allocations, and CPU offloading for serialization.
    """

    def __init__(self, hidden=64, epochs=10, lr=1e-3, batch=config.MODEL_FFN_BATCH, random_state=config.RANDOM_STATE):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.random_state = random_state
        self.model_ = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    def _build_model(self, in_dim, output_dim, is_classifier):
        layers = [
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, output_dim)
        ]
        if is_classifier:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers).to(self.device)

    def _fit_shared(self, X, y, loss_fn):
        if hasattr(X, "toarray"): X = X.toarray()

        # Use np.asarray with float32 to allow zero-copy if input is already float32
        X = np.asarray(X, dtype=np.float32)

        self.in_dim_ = X.shape[1]
        torch.manual_seed(self.random_state)

        self.model_ = self._build_model(self.in_dim_, 1, isinstance(loss_fn, nn.BCELoss))
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        try:
            X_t = torch.as_tensor(X, device=self.device)
            y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        except RuntimeError:
            X_t = torch.from_numpy(X)
            y_t = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

        n_samples = len(X)
        n_batches = (n_samples + self.batch - 1) // self.batch

        self.model_.train()
        on_gpu = (X_t.device.type == self.device.type) and (self.device.type != 'cpu')

        for _ in range(self.epochs):
            perm = torch.randperm(n_samples, device=self.device if on_gpu else 'cpu')

            for i in range(n_batches):
                start = i * self.batch
                end = start + self.batch
                indices = perm[start:end]

                if on_gpu:
                    xb, yb = X_t[indices], y_t[indices]
                else:
                    xb = X_t[indices].to(self.device, non_blocking=True)
                    yb = y_t[indices].to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        # Move model to CPU before serialization/pickling.
        self.model_.cpu()
        del opt
        torch.cuda.empty_cache()
        return self

    def _predict_shared(self, X):
        if hasattr(X, "toarray"): X = X.toarray()
        # Zero-copy check
        X = np.asarray(X, dtype=np.float32)

        # Move to GPU for inference
        self.model_.to(self.device)
        self.model_.eval()

        n_samples = len(X)
        result = np.empty(n_samples, dtype=np.float32)
        inf_batch = self.batch * 4

        with torch.inference_mode():
            try:
                X_all = torch.as_tensor(X, device=self.device)
                preds = self.model_(X_all)
                return preds.ravel().cpu().numpy()
            except RuntimeError:
                pass

            for i in range(0, n_samples, inf_batch):
                end = min(i + inf_batch, n_samples)
                batch_np = X[i:end]
                batch_t = torch.as_tensor(batch_np, device=self.device)
                preds = self.model_(batch_t)
                result[i:end] = preds.ravel().cpu().numpy()

        return result


class TorchMLPClassifier(TorchMLPBase, ClassifierMixin):
    def fit(self, X, y):
        return self._fit_shared(X, y, nn.BCELoss())

    def predict_proba(self, X):
        p = self._predict_shared(X)
        out = np.empty((len(p), 2), dtype=np.float32)
        out[:, 0] = 1.0 - p
        out[:, 1] = p
        return out

    def predict(self, X):
        return (self._predict_shared(X) >= 0.5).astype(int)


class TorchMLPRegressor(TorchMLPBase, RegressorMixin):
    def fit(self, X, y):
        return self._fit_shared(X, y, nn.MSELoss())

    def predict(self, X):
        return self._predict_shared(X)


def build_classifiers(random_state=config.RANDOM_STATE):
    # Ensure single-threaded internal execution for models
    n_jobs = 1
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    svm_approx = Pipeline([
        ("nystroem", Nystroem(gamma=0.2, random_state=random_state, n_components=300, n_jobs=n_jobs)),
        ("sgd_svm",
         SGDClassifier(loss='hinge', alpha=1e-4, max_iter=1000, early_stopping=True, random_state=random_state,
                       n_jobs=n_jobs))
    ])

    return {
        config.MODEL_LOGREG: LogisticRegression(max_iter=config.LOGREG_MAX_ITER, solver=config.LOGREG_SOLVER,
                                                random_state=random_state, n_jobs=n_jobs),
        config.MODEL_SVM: svm_approx,
        config.MODEL_RF: RandomForestClassifier(n_estimators=config.RF_ESTIMATORS, random_state=random_state,
                                                n_jobs=n_jobs),
        config.MODEL_XGB: xgb.XGBClassifier(n_estimators=config.XGB_ESTIMATORS, max_depth=6, tree_method="hist",
                                            device=xgb_device, eval_metric="logloss", random_state=random_state,
                                            n_jobs=n_jobs),
        config.MODEL_FFN: TorchMLPClassifier(hidden=128, epochs=10, lr=1e-3, batch=config.MODEL_FFN_BATCH,
                                             random_state=random_state)
    }


def build_regressors(random_state=config.RANDOM_STATE):
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    n_jobs = 1

    svr_approx = Pipeline([
        ("nystroem", Nystroem(gamma=0.2, random_state=random_state, n_components=300, n_jobs=n_jobs)),
        ("sgd_svr", SGDRegressor(max_iter=1000, early_stopping=True, random_state=random_state))
    ])

    return {
        config.MODEL_LINREG: LinearRegression(n_jobs=n_jobs),
        config.MODEL_SVR: svr_approx,
        config.MODEL_RF_REG: RandomForestRegressor(n_estimators=config.RF_ESTIMATORS, random_state=random_state,
                                                   n_jobs=n_jobs),
        config.MODEL_XGB_REG: xgb.XGBRegressor(n_estimators=config.XGB_ESTIMATORS, max_depth=8, tree_method="hist",
                                               device=xgb_device, random_state=random_state, n_jobs=n_jobs),
        config.MODEL_FFN_REG: TorchMLPRegressor(hidden=128, epochs=10, lr=1e-3, batch=config.MODEL_FFN_BATCH,
                                                random_state=random_state)
    }


def evaluate_classifier(pipe, X_train, y_train, X_test, y_test, refit=True):
    if refit: pipe.fit(X_train, y_train)
    if hasattr(pipe, "predict_proba"):
        preds = pipe.predict_proba(X_test)[:, 1]
    else:
        try:
            preds = pipe.decision_function(X_test)
        except:
            preds = pipe.predict(X_test)
    return roc_auc_score(y_test, preds), None


def evaluate_regressor(pipe, X_train, y_train, X_test, y_test, refit=True):
    if refit: pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return root_mean_squared_error(y_test, preds), None