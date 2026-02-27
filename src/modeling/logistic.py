from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coef: np.ndarray


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logit(X: pd.DataFrame, y: pd.Series, C: float = 1.0) -> LogisticModel:
    X_values = np.asarray(X, dtype=float)
    y_values = np.asarray(y, dtype=float)

    mean = X_values.mean(axis=0)
    scale = X_values.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    X_scaled = (X_values - mean) / scale

    reg_strength = 1.0 / C if C > 0 else 1.0
    sample_count = max(len(y_values), 1)
    params = np.zeros(X_scaled.shape[1] + 1, dtype=float)
    learning_rate = 0.1

    for _ in range(400):
        intercept = params[0]
        coef = params[1:]
        probs = _sigmoid(intercept + X_scaled @ coef)
        diff = probs - y_values
        grad_intercept = diff.mean()
        grad_coef = (X_scaled.T @ diff) / sample_count + (reg_strength * coef) / sample_count
        grad = np.concatenate([[grad_intercept], grad_coef])
        params -= learning_rate * grad

    return LogisticModel(mean=mean, scale=scale, intercept=float(params[0]), coef=np.asarray(params[1:], dtype=float))


def predict_proba(model: LogisticModel, X: pd.DataFrame) -> pd.Series:
    X_values = np.asarray(X, dtype=float)
    X_scaled = (X_values - model.mean) / model.scale
    probs = _sigmoid(model.intercept + X_scaled @ model.coef)
    return pd.Series(probs, index=X.index, name="p_up")
