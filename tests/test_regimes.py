import pandas as pd

from src.regimes.vol_regime import apply_regime_labels, fit_regime_thresholds


def test_regime_thresholds_are_fit_on_train_only() -> None:
    train_df = pd.DataFrame({"spy_rv_20d": [0.10, 0.12, 0.14, 0.16, 0.18]})
    test_df = pd.DataFrame({"spy_rv_20d": [0.11, 0.50]})

    thresholds = fit_regime_thresholds(train_df, low_pct=0.30, high_pct=0.70)
    labeled = apply_regime_labels(test_df, thresholds)

    assert thresholds["high"] < 0.50
    assert labeled.loc[0, "regime"] in {"low", "mid", "high"}
    assert labeled.loc[1, "regime"] == "high"
