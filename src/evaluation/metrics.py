from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float:
    truth = pd.Series(y_true).astype(int).to_numpy(dtype=float)
    prob = pd.Series(y_prob).clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)
    return float(-(truth * np.log(prob) + (1.0 - truth) * np.log(1.0 - prob)).mean())


def compute_brier(y_true: pd.Series, y_prob: pd.Series) -> float:
    truth = pd.Series(y_true).astype(int).to_numpy(dtype=float)
    prob = pd.Series(y_prob).clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)
    return float(np.mean((prob - truth) ** 2))


def forecast_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in prediction_df.groupby("model_name"):
        rows.append(
            {
                "model_name": model_name,
                "log_loss": compute_log_loss(group["y_up"], group["p_up"]),
                "brier_score": compute_brier(group["y_up"], group["p_up"]),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("model_name")


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def portfolio_metrics(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, cost_bps), group in portfolio_df.groupby(["model_name", "cost_bps"]):
        ret = group["net_return"].fillna(0.0)
        ann_return = float(ret.mean() * 252)
        ann_vol = float(ret.std(ddof=0) * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        rows.append(
            {
                "model_name": model_name,
                "cost_bps": cost_bps,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": _max_drawdown(ret),
                "avg_turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "cost_bps"])


def portfolio_summary_table(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    return portfolio_metrics(portfolio_df)
