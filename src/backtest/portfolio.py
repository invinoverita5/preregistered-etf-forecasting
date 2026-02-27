from __future__ import annotations

import numpy as np
import pandas as pd


def probabilities_to_signal(probs: pd.Series, base_rate: pd.Series) -> pd.Series:
    return 2.0 * (probs - base_rate)


def build_positions(
    prediction_df: pd.DataFrame,
    vol_col: str = "rv_20d",
    center_col: str = "train_base_rate",
    max_abs_weight: float = 1.0,
) -> pd.DataFrame:
    required = {"date", "symbol", "model_name", "p_up", "fwd_ret_1d", vol_col, center_col}
    missing = required.difference(prediction_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for build_positions: {sorted(missing)}")

    out = prediction_df.copy()
    out["signal"] = probabilities_to_signal(out["p_up"], out[center_col])
    safe_vol = out[vol_col].replace(0, np.nan)
    out["weight"] = (out["signal"] / safe_vol).clip(-max_abs_weight, max_abs_weight).fillna(0.0)
    return out.loc[:, ["date", "symbol", "model_name", "weight", "fwd_ret_1d"]]


def _return_matrix(feature_df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    panel = (
        feature_df.loc[feature_df["symbol"].isin(symbols), ["date", "symbol", "fwd_ret_1d"]]
        .drop_duplicates()
        .sort_values(["date", "symbol"])
    )
    matrix = panel.pivot(index="date", columns="symbol", values="fwd_ret_1d").reindex(columns=symbols).sort_index()
    return matrix.fillna(0.0)


def _drifting_long_only_path(
    feature_df: pd.DataFrame,
    symbols: list[str],
    rebalance_dates: pd.DatetimeIndex,
    model_name: str,
) -> pd.DataFrame:
    returns = _return_matrix(feature_df, symbols)
    if returns.empty:
        return pd.DataFrame(columns=["date", "symbol", "weight", "model_name"])

    rebalance_set = set(pd.to_datetime(rebalance_dates))
    target_weights = np.full(len(symbols), 1.0 / len(symbols), dtype=float)
    current = target_weights.copy()
    rows: list[dict] = []

    for idx, date in enumerate(returns.index):
        if idx == 0 or date in rebalance_set:
            current = target_weights.copy()

        for symbol, weight in zip(symbols, current):
            rows.append({"date": date, "symbol": symbol, "weight": float(weight), "model_name": model_name})

        day_returns = returns.loc[date].to_numpy(dtype=float)
        gross_return = float(np.dot(current, day_returns))
        denom = 1.0 + gross_return
        if abs(denom) < 1e-12:
            current = target_weights.copy()
        else:
            current = current * (1.0 + day_returns) / denom

    return pd.DataFrame(rows)


def build_buy_hold_positions(feature_df: pd.DataFrame, symbols: list[str], model_name: str) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(feature_df.loc[feature_df["symbol"].isin(symbols), "date"].drop_duplicates()))
    first_date = dates[:1]
    return _drifting_long_only_path(feature_df, symbols, rebalance_dates=first_date, model_name=model_name)


def build_equal_weight_positions(
    feature_df: pd.DataFrame,
    symbols: list[str],
    rebalance: str,
    model_name: str,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(feature_df.loc[feature_df["symbol"].isin(symbols), "date"].drop_duplicates()))
    if rebalance == "D":
        rebalance_dates = dates
    elif rebalance == "M":
        tmp = pd.DataFrame({"date": dates})
        tmp["ym"] = tmp["date"].dt.to_period("M")
        rebalance_dates = pd.DatetimeIndex(tmp.groupby("ym")["date"].min().sort_values())
    else:
        raise ValueError("rebalance must be 'D' or 'M'")

    return _drifting_long_only_path(feature_df, symbols, rebalance_dates=rebalance_dates, model_name=model_name)


def build_vol_target_only_positions(
    feature_df: pd.DataFrame,
    symbols: list[str],
    vol_col: str,
    max_abs_weight: float,
    model_name: str,
    gross_cap: float = 1.0,
) -> pd.DataFrame:
    df = (
        feature_df.loc[feature_df["symbol"].isin(symbols), ["date", "symbol", vol_col]]
        .drop_duplicates()
        .sort_values(["date", "symbol"])
        .copy()
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "weight", "model_name"])

    eps = 1e-12
    df["raw"] = 1.0 / df[vol_col].clip(lower=eps)
    gross = df.groupby("date")["raw"].transform("sum").clip(lower=eps)
    df["weight"] = (df["raw"] / gross) * gross_cap
    df["weight"] = df["weight"].clip(0.0, max_abs_weight)
    gross2 = df.groupby("date")["weight"].transform("sum").clip(lower=eps)
    df["weight"] = (df["weight"] / gross2) * gross_cap
    df["model_name"] = model_name
    return df.loc[:, ["date", "symbol", "weight", "model_name"]]
