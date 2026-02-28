from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.targets import build_binary_target
from src.utils.config import load_config


def build_feature_frame(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    df = prices.sort_values(["symbol", "date"]).copy()
    grouped = df.groupby("symbol", group_keys=False)

    df["ret_1d"] = grouped["adj_close"].pct_change()
    df["ret_5d"] = grouped["adj_close"].pct_change(5)
    df["ret_20d"] = grouped["adj_close"].pct_change(20)
    df["rv_5d"] = grouped["ret_1d"].rolling(5).std().reset_index(level=0, drop=True) * np.sqrt(252)
    df["rv_20d"] = grouped["ret_1d"].rolling(20).std().reset_index(level=0, drop=True) * np.sqrt(252)
    df["vol_ratio"] = df["rv_5d"] / df["rv_20d"]

    spy_context = (
        df.loc[df["symbol"] == "SPY", ["date", "ret_20d", "rv_20d"]]
        .rename(columns={"ret_20d": "spy_ret_20d", "rv_20d": "spy_rv_20d"})
        .copy()
    )
    feature_df = df.merge(spy_context, on="date", how="left")
    feature_df = build_binary_target(feature_df, horizon=horizon_days)
    return feature_df


def main() -> None:
    config = load_config("config/experiment_mvp.yaml")
    price_path = Path("data/processed/prices.parquet")
    prices = pd.read_parquet(price_path)
    feature_df = build_feature_frame(prices, horizon_days=int(config["target"]["horizon_days"]))
    feature_df.to_parquet("data/processed/features.parquet", index=False)


if __name__ == "__main__":
    main()
