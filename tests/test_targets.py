import math

import pandas as pd

from src.features.targets import build_binary_target


def test_build_binary_target_keeps_daily_backtest_return_and_uses_horizon_target() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "adj_close": [100.0, 102.0, 101.0, 104.0],
        }
    )

    out = build_binary_target(df, horizon=2)

    assert math.isclose(out.loc[0, "fwd_ret_1d"], 0.02, rel_tol=1e-9)
    assert out.loc[0, "y_up"] == 1
    assert math.isclose(out.loc[0, "fwd_ret_target"], 0.01, rel_tol=1e-9)
    assert math.isclose(out.loc[1, "fwd_ret_1d"], -0.009803921568627416, rel_tol=1e-9)
    assert math.isclose(out.loc[1, "fwd_ret_target"], 0.019607843137254832, rel_tol=1e-9)
    assert out.loc[1, "y_up"] == 1
    assert out.loc[0, "target_horizon_days"] == 2
    assert pd.isna(out.loc[2, "y_up"])
    assert pd.isna(out.loc[3, "y_up"])
