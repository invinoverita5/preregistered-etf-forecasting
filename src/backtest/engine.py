from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.costs import apply_turnover_costs
from src.backtest.portfolio import (
    build_buy_hold_positions,
    build_equal_weight_positions,
    build_positions,
    build_vol_target_only_positions,
)
from src.utils.config import load_config


def _oos_feature_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    panel = predictions.loc[predictions["model_name"] == "base_rate", ["date", "symbol", "fwd_ret_1d", "rv_20d"]]
    return panel.drop_duplicates().sort_values(["date", "symbol"]).reset_index(drop=True)


def run_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    experiment = load_config("config/experiment_mvp.yaml")
    universe = load_config("config/universe.yaml")
    symbols = universe["symbols"]

    model_positions = build_positions(
        prediction_df=predictions,
        vol_col=experiment["portfolio"]["vol_col"],
        center_col="train_base_rate",
        max_abs_weight=float(experiment["portfolio"]["max_abs_weight"]),
        rebalance=experiment["portfolio"].get("rebalance", "D"),
    )
    oos_panel = _oos_feature_panel(predictions)

    baseline_positions = [
        build_buy_hold_positions(oos_panel, symbols, model_name="buy_hold_equal_weight"),
        build_equal_weight_positions(oos_panel, symbols, rebalance="D", model_name="equal_weight_daily"),
        build_equal_weight_positions(oos_panel, symbols, rebalance="M", model_name="equal_weight_monthly"),
        build_vol_target_only_positions(
            oos_panel,
            symbols,
            vol_col=experiment["portfolio"]["vol_col"],
            max_abs_weight=float(experiment["portfolio"]["max_abs_weight"]),
            model_name="vol_target_only",
        ),
    ]
    for symbol in symbols:
        baseline_positions.append(build_buy_hold_positions(oos_panel, [symbol], model_name=f"buy_hold_{symbol}"))

    baseline_position_df = pd.concat(baseline_positions, ignore_index=True)
    baseline_position_df = baseline_position_df.merge(
        oos_panel.loc[:, ["date", "symbol", "fwd_ret_1d"]],
        on=["date", "symbol"],
        how="left",
        validate="many_to_one",
    )

    position_df = pd.concat([model_positions, baseline_position_df], ignore_index=True).sort_values(
        ["model_name", "date", "symbol"]
    )

    daily_frames = [
        apply_turnover_costs(position_df, cost_bps=float(cost_bps))
        for cost_bps in experiment["costs"]["bps"]
    ]
    return pd.concat(daily_frames, ignore_index=True).sort_values(["model_name", "cost_bps", "date"])


def main() -> None:
    prediction_path = Path("data/processed/predictions.parquet")
    predictions = pd.read_parquet(prediction_path)
    portfolio = run_backtest(predictions)
    portfolio.to_parquet("data/processed/portfolio_daily.parquet", index=False)


if __name__ == "__main__":
    main()
