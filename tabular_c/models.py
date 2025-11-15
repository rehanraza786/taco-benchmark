import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from . import config

def build_preprocessor(X: pd.DataFrame, numeric_strategy=config.DEFAULT_IMPUTE_STRATEGY):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_tf = Pipeline([("imputer", SimpleImputer(strategy=numeric_strategy)), ("scaler", StandardScaler())])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(transformers=[("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])
    return pre

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=64, epochs=10, lr=1e-3, batch=256, random_state=config.RANDOM_STATE):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.random_state = random_state
        self.model_ = None
        self.in_dim_ = None

    def _to_numpy(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        X = self._to_numpy(X) 
        y = np.asarray(y, dtype=np.float32).reshape(-1,1)
        self.in_dim_ = X.shape[1]
        torch.manual_seed(self.random_state)
        self.model_ = nn.Sequential(
            nn.Linear(self.in_dim_, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _ in range(self.epochs):
            for i in range(0, len(X), self.batch):
                xb = X_t[i:i+self.batch]
                yb = y_t[i:i+self.batch]
                opt.zero_grad()
                preds = self.model_(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        X = self._to_numpy(X) 
        with torch.no_grad():
            p1 = self.model_(torch.from_numpy(X)).numpy().reshape(-1)
        return np.vstack([1-p1, p1]).T

    def predict(self, X):
        X = self._to_numpy(X) 
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=64, epochs=10, lr=1e-3, batch=256, random_state=config.RANDOM_STATE):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.random_state = random_state
        self.model_ = None
        self.in_dim_ = None

    def _to_numpy(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        X = self._to_numpy(X) 
        y = np.asarray(y, dtype=np.float32).reshape(-1,1)
        self.in_dim_ = X.shape[1]
        torch.manual_seed(self.random_state)
        self.model_ = nn.Sequential(
            nn.Linear(self.in_dim_, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _ in range(self.epochs):
            for i in range(0, len(X), self.batch):
                xb = X_t[i:i+self.batch]
                yb = y_t[i:i+self.batch]
                opt.zero_grad()
                preds = self.model_(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        X = self._to_numpy(X)
        with torch.no_grad():
            p = self.model_(torch.from_numpy(X)).numpy().reshape(-1)
        return p

def build_classifiers(random_state=config.RANDOM_STATE):
    return {
        config.MODEL_LOGREG: LogisticRegression(max_iter=1000, random_state=random_state),
        config.MODEL_SVM: SVC(probability=True, random_state=random_state),
        config.MODEL_RF: RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        config.MODEL_XGB: xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss", random_state=random_state, n_jobs=-1
        ),
        config.MODEL_FFN: TorchMLPClassifier(hidden=128, epochs=10, lr=1e-3, batch=512, random_state=random_state)
    }

def build_regressors(random_state=config.RANDOM_STATE):
    return {
        config.MODEL_LINREG: LinearRegression(),
        config.MODEL_SVR: SVR(),
        config.MODEL_RF_REG: RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        config.MODEL_XGB_REG: xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=random_state, n_jobs=-1
        ),
        config.MODEL_FFN_REG: TorchMLPRegressor(hidden=128, epochs=10, lr=1e-3, batch=512, random_state=random_state)
    }

def evaluate_classifier(pipeline, X_train, y_train, X_test, y_test, refit=True):
    if refit:
        pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    return auc, proba

def evaluate_regressor(pipeline, X_train, y_train, X_test, y_test, refit=True):
    if refit:
        pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse, preds