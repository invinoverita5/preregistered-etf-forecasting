from __future__ import annotations

import pandas as pd


def generate_walkforward_splits(
    dates: pd.Series | pd.Index,
    train_years: int = 3,
    test_months: int = 6,
    step_months: int = 3,
) -> list[dict]:
    unique_dates = pd.DatetimeIndex(pd.Series(dates).dropna().sort_values().unique())
    if unique_dates.empty:
        return []

    splits: list[dict] = []
    start_idx = 0

    while start_idx < len(unique_dates):
        train_start = unique_dates[start_idx]
        train_cutoff = train_start + pd.DateOffset(years=train_years)
        test_cutoff = train_cutoff + pd.DateOffset(months=test_months)

        train_dates = unique_dates[(unique_dates >= train_start) & (unique_dates < train_cutoff)]
        test_dates = unique_dates[(unique_dates >= train_cutoff) & (unique_dates < test_cutoff)]

        if train_dates.empty or test_dates.empty:
            break

        splits.append(
            {
                "fold_id": len(splits),
                "train_start": train_dates.min(),
                "train_end": train_dates.max(),
                "test_start": test_dates.min(),
                "test_end": test_dates.max(),
                "train_dates": train_dates,
                "test_dates": test_dates,
            }
        )

        next_start = train_start + pd.DateOffset(months=step_months)
        start_idx = unique_dates.searchsorted(next_start)

    return splits
