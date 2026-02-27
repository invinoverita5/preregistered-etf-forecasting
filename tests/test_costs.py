import pandas as pd

from src.backtest.costs import add_turnover, apply_turnover_costs


def test_turnover_and_cost_increase_when_weights_flip_sign() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["SPY", "SPY"],
            "model_name": ["logit_plain", "logit_plain"],
            "weight": [0.5, -0.5],
            "fwd_ret_1d": [0.01, -0.01],
        }
    )

    daily = add_turnover(df)
    assert daily.loc[1, "turnover"] > 1.0

    cost_5 = apply_turnover_costs(df, cost_bps=5.0)
    cost_10 = apply_turnover_costs(df, cost_bps=10.0)

    assert cost_10["cost_return"].sum() > cost_5["cost_return"].sum()
