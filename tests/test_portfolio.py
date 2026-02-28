import math

import pandas as pd

from src.backtest.costs import add_turnover
from src.backtest.portfolio import build_positions


def test_build_positions_centers_signal_on_train_base_rate() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["SPY", "SPY"],
            "model_name": ["base_rate", "logit_plain"],
            "p_up": [0.54, 0.59],
            "train_base_rate": [0.54, 0.54],
            "fwd_ret_1d": [0.01, -0.01],
            "rv_20d": [0.20, 0.20],
        }
    )

    out = build_positions(df, vol_col="rv_20d", center_col="train_base_rate", max_abs_weight=1.0)

    assert math.isclose(out.loc[0, "weight"], 0.0, rel_tol=1e-9)
    assert math.isclose(out.loc[1, "weight"], 0.5, rel_tol=1e-9)


def test_build_positions_weekly_rebalance_holds_between_rebalance_dates() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-08"]),
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "model_name": ["logit_plain"] * 4,
            "p_up": [0.75, 0.25, 0.75, 0.25],
            "train_base_rate": [0.50] * 4,
            "fwd_ret_1d": [0.0, 0.0, 0.0, 0.0],
            "rv_20d": [1.0] * 4,
        }
    )

    daily = build_positions(df, vol_col="rv_20d", center_col="train_base_rate", max_abs_weight=1.0, rebalance="D")
    weekly = build_positions(df, vol_col="rv_20d", center_col="train_base_rate", max_abs_weight=1.0, rebalance="W")

    assert math.isclose(weekly.loc[0, "weight"], 0.5, rel_tol=1e-9)
    assert math.isclose(weekly.loc[1, "weight"], 0.5, rel_tol=1e-9)
    assert math.isclose(weekly.loc[2, "weight"], 0.5, rel_tol=1e-9)
    assert math.isclose(weekly.loc[3, "weight"], -0.5, rel_tol=1e-9)
    assert add_turnover(weekly)["turnover"].sum() < add_turnover(daily)["turnover"].sum()
