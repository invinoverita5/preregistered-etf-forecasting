from __future__ import annotations

import numpy as np
import pandas as pd


def add_turnover(position_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "model_name", "weight", "fwd_ret_1d"}
    missing = required.difference(position_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for turnover calculation: {sorted(missing)}")

    summaries: list[dict] = []
    sorted_df = position_df.sort_values(["model_name", "date", "symbol"]).copy()

    for model_name, group in sorted_df.groupby("model_name"):
        weight_frame = group.pivot(index="date", columns="symbol", values="weight").sort_index().fillna(0.0)
        return_frame = (
            group.pivot(index="date", columns="symbol", values="fwd_ret_1d")
            .reindex(index=weight_frame.index, columns=weight_frame.columns)
            .fillna(0.0)
        )

        pre_trade = np.zeros(len(weight_frame.columns), dtype=float)
        for date in weight_frame.index:
            target = weight_frame.loc[date].to_numpy(dtype=float)
            day_returns = return_frame.loc[date].to_numpy(dtype=float)
            turnover = float(np.abs(target - pre_trade).sum())
            gross_return = float(np.dot(target, day_returns))
            summaries.append(
                {
                    "date": date,
                    "model_name": model_name,
                    "gross_return": gross_return,
                    "turnover": turnover,
                    "gross_exposure": float(np.abs(target).sum()),
                    "net_exposure": float(target.sum()),
                }
            )

            denom = 1.0 + gross_return
            if abs(denom) < 1e-12:
                pre_trade = np.zeros_like(target)
            else:
                pre_trade = target * (1.0 + day_returns) / denom

    return pd.DataFrame(summaries).sort_values(["model_name", "date"]).reset_index(drop=True)


def apply_turnover_costs(position_df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    daily = add_turnover(position_df)
    daily["cost_return"] = daily["turnover"] * (cost_bps / 10000.0)
    daily["net_return"] = daily["gross_return"] - daily["cost_return"]
    daily["cost_bps"] = cost_bps
    return daily
