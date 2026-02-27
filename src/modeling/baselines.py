from __future__ import annotations

import numpy as np
import pandas as pd


def predict_base_rate(train_y: pd.Series, rows: int) -> np.ndarray:
    base_rate = float(train_y.dropna().mean())
    return np.full(rows, base_rate)


def predict_momentum_sign(df: pd.DataFrame, signal_col: str = "ret_20d") -> np.ndarray:
    signal = df[signal_col].fillna(0.0)
    return np.where(signal > 0, 1.0, np.where(signal < 0, 0.0, 0.5))
