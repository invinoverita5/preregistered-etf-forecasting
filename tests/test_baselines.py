import pandas as pd

from src.backtest.costs import add_turnover
from src.backtest.portfolio import (
    build_buy_hold_positions,
    build_equal_weight_positions,
    build_vol_target_only_positions,
)


def _sample_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-02-01",
            "2024-02-02",
        ]
    )
    rows = []
    returns = {
        "SPY": [0.02, -0.01, 0.03, -0.02],
        "TLT": [-0.01, 0.02, -0.02, 0.01],
    }
    vols = {
        "SPY": [0.10, 0.15, 0.30, 0.20],
        "TLT": [0.20, 0.25, 0.40, 0.30],
    }
    for symbol in ["SPY", "TLT"]:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "fwd_ret_1d": returns[symbol][idx],
                    "rv_20d": vols[symbol][idx],
                }
            )
    return pd.DataFrame(rows)


def test_equal_weight_monthly_turnover_is_lower_than_daily() -> None:
    panel = _sample_panel()
    symbols = ["SPY", "TLT"]

    daily = build_equal_weight_positions(panel, symbols, "D", "equal_weight_daily").merge(
        panel[["date", "symbol", "fwd_ret_1d"]], on=["date", "symbol"], how="left"
    )
    monthly = build_equal_weight_positions(panel, symbols, "M", "equal_weight_monthly").merge(
        panel[["date", "symbol", "fwd_ret_1d"]], on=["date", "symbol"], how="left"
    )

    daily_turnover = add_turnover(daily)["turnover"].sum()
    monthly_turnover = add_turnover(monthly)["turnover"].sum()
    assert monthly_turnover <= daily_turnover


def test_buy_hold_has_no_turnover_after_initial_allocation() -> None:
    panel = _sample_panel()
    positions = build_buy_hold_positions(panel, ["SPY", "TLT"], "buy_hold_equal_weight").merge(
        panel[["date", "symbol", "fwd_ret_1d"]], on=["date", "symbol"], how="left"
    )

    turnover = add_turnover(positions)
    assert turnover.iloc[0]["turnover"] > 0.0
    assert turnover.iloc[1:]["turnover"].abs().sum() < 1e-10


def test_vol_target_only_shrinks_weights_when_vol_rises() -> None:
    panel = _sample_panel()
    positions = build_vol_target_only_positions(
        panel,
        symbols=["SPY", "TLT"],
        vol_col="rv_20d",
        max_abs_weight=1.0,
        model_name="vol_target_only",
    )

    spy_rows = positions[positions["symbol"] == "SPY"].sort_values("date").reset_index(drop=True)
    assert spy_rows.loc[2, "weight"] < spy_rows.loc[0, "weight"]
