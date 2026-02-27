import math

import pandas as pd

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
