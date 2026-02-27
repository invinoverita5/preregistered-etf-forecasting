import math

import pandas as pd

from src.backtest.costs import add_turnover


def test_backtest_applies_target_weight_to_same_row_forward_return() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["SPY", "SPY"],
            "model_name": ["timing_test", "timing_test"],
            "weight": [1.0, 0.0],
            "fwd_ret_1d": [0.10, -0.50],
        }
    )

    daily = add_turnover(df)

    assert math.isclose(daily.loc[0, "gross_return"], 0.10, rel_tol=1e-9)
    assert math.isclose(daily.loc[1, "gross_return"], 0.0, rel_tol=1e-9)
    assert math.isclose(daily.loc[1, "turnover"], 1.0, rel_tol=1e-9)
