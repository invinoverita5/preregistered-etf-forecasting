from __future__ import annotations

import pandas as pd


def fit_regime_thresholds(
    train_df: pd.DataFrame,
    vol_col: str = "spy_rv_20d",
    low_pct: float = 0.30,
    high_pct: float = 0.70,
) -> dict:
    series = train_df[vol_col].dropna()
    if series.empty:
        raise ValueError(f"No values available in {vol_col} for threshold fitting.")

    return {
        "vol_col": vol_col,
        "low": float(series.quantile(low_pct)),
        "high": float(series.quantile(high_pct)),
    }


def apply_regime_labels(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    vol_col = thresholds["vol_col"]
    low = thresholds["low"]
    high = thresholds["high"]

    out = df.copy()
    out["regime"] = "mid"
    out.loc[out[vol_col] < low, "regime"] = "low"
    out.loc[out[vol_col] > high, "regime"] = "high"
    return out
