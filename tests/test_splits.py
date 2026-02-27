import pandas as pd

from src.utils.splits import generate_walkforward_splits


def test_generate_walkforward_splits_is_time_ordered_and_non_overlapping() -> None:
    dates = pd.date_range("2020-01-01", periods=1800, freq="B")

    splits = generate_walkforward_splits(dates, train_years=3, test_months=6, step_months=3)

    assert splits
    first = splits[0]
    assert first["train_end"] < first["test_start"]
    assert set(first["train_dates"]).isdisjoint(set(first["test_dates"]))
