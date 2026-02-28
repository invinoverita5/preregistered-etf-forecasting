from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.baselines import predict_base_rate, predict_momentum_sign
from src.modeling.logistic import fit_logit, predict_proba
from src.utils.config import load_config
from src.utils.splits import generate_walkforward_splits

BASE_FEATURE_COLS = [
    "ret_5d",
    "ret_20d",
    "rv_20d",
    "vol_ratio",
    "spy_ret_20d",
    "spy_rv_20d",
]

SYMBOL_REFERENCE = "SPY"
SYMBOL_DUMMY_COLS = [
    "symbol_QQQ",
    "symbol_IWM",
    "symbol_TLT",
    "symbol_HYG",
    "symbol_GLD",
]


def _one_hot_symbols(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["symbol"], prefix="symbol", dtype=float)
    dummies = dummies.drop(columns=[f"symbol_{SYMBOL_REFERENCE}"], errors="ignore")
    return dummies.reindex(columns=SYMBOL_DUMMY_COLS, fill_value=0.0)


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[BASE_FEATURE_COLS].copy()
    X = pd.concat([X, _one_hot_symbols(df)], axis=1)
    return X.fillna(0.0)


def fixed_matrix_columns() -> list[str]:
    return BASE_FEATURE_COLS + SYMBOL_DUMMY_COLS


def shuffle_targets(
    train_df: pd.DataFrame,
    y_col: str = "y_up",
    by_symbol: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = train_df.copy()

    if by_symbol and "symbol" in shuffled.columns:
        shuffled[y_col] = (
            shuffled.groupby("symbol", group_keys=False)[y_col]
            .transform(lambda s: rng.permutation(s.to_numpy()))
            .astype(int)
        )
    else:
        shuffled[y_col] = rng.permutation(shuffled[y_col].to_numpy()).astype(int)

    return shuffled


def _train_base_rate_summary(train_df: pd.DataFrame) -> tuple[float, pd.Series]:
    overall = float(train_df["y_up"].astype(int).mean())
    by_symbol = train_df.groupby("symbol")["y_up"].mean().astype(float)
    return overall, by_symbol


def _prediction_frame(
    test_df: pd.DataFrame,
    split: dict,
    model_name: str,
    preds: np.ndarray | pd.Series,
    train_base_rate: float,
    train_symbol_rates: pd.Series,
) -> pd.DataFrame:
    fold_rows = test_df.loc[:, ["date", "symbol", "y_up", "fwd_ret_target", "fwd_ret_1d", "rv_20d", "target_horizon_days"]].copy()
    fold_rows["fold_id"] = split["fold_id"]
    fold_rows["train_start"] = split["train_start"]
    fold_rows["train_end"] = split["train_end"]
    fold_rows["test_start"] = split["test_start"]
    fold_rows["test_end"] = split["test_end"]
    fold_rows["train_base_rate"] = train_base_rate
    fold_rows["train_symbol_base_rate"] = fold_rows["symbol"].map(train_symbol_rates).astype(float)
    fold_rows["model_name"] = model_name
    fold_rows["p_up"] = np.asarray(preds, dtype=float)
    return fold_rows


def run_walkforward(features: pd.DataFrame) -> pd.DataFrame:
    config = load_config("config/experiment_mvp.yaml")
    splits = generate_walkforward_splits(
        dates=features["date"],
        train_years=config["walkforward"]["train_years"],
        test_months=config["walkforward"]["test_months"],
        step_months=config["walkforward"]["step_months"],
    )

    rows = []
    clean = features.dropna(subset=BASE_FEATURE_COLS + ["y_up", "fwd_ret_target", "fwd_ret_1d"]).copy()
    clean["y_up"] = clean["y_up"].astype(int)

    for split in splits:
        train_mask = clean["date"].isin(split["train_dates"])
        test_mask = clean["date"].isin(split["test_dates"])
        train_df = clean.loc[train_mask].copy()
        test_df = clean.loc[test_mask].copy()

        shuffled_train_df = shuffle_targets(train_df, seed=42 + int(split["fold_id"]))
        train_base_rate, train_symbol_rates = _train_base_rate_summary(train_df)

        base_preds = predict_base_rate(train_df["y_up"].astype(int), len(test_df))
        mom_preds = predict_momentum_sign(test_df)

        for model_name, preds in [
            ("base_rate", base_preds),
            ("mom_sign_20d", mom_preds),
        ]:
            rows.append(
                _prediction_frame(
                    test_df=test_df,
                    split=split,
                    model_name=model_name,
                    preds=preds,
                    train_base_rate=train_base_rate,
                    train_symbol_rates=train_symbol_rates,
                )
            )

        X_train = build_design_matrix(train_df)
        X_test = build_design_matrix(test_df)
        model = fit_logit(X_train, train_df["y_up"].astype(int))
        probs = predict_proba(model, X_test)
        rows.append(
            _prediction_frame(
                test_df=test_df,
                split=split,
                model_name="logit_plain",
                preds=probs,
                train_base_rate=train_base_rate,
                train_symbol_rates=train_symbol_rates,
            )
        )

        X_train_shuffled = build_design_matrix(shuffled_train_df)
        shuffled_model = fit_logit(X_train_shuffled, shuffled_train_df["y_up"].astype(int))
        shuffled_probs = predict_proba(shuffled_model, X_test)
        rows.append(
            _prediction_frame(
                test_df=test_df,
                split=split,
                model_name="logit_plain_shuffled",
                preds=shuffled_probs,
                train_base_rate=train_base_rate,
                train_symbol_rates=train_symbol_rates,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "y_up",
                "fwd_ret_target",
                "fwd_ret_1d",
                "rv_20d",
                "target_horizon_days",
                "fold_id",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "train_base_rate",
                "train_symbol_base_rate",
                "model_name",
                "p_up",
            ]
        )

    predictions = pd.concat(rows, ignore_index=True).sort_values(["model_name", "date", "symbol", "fold_id"])
    # The 3m step / 6m test protocol overlaps test windows. Keep the earliest OOS prediction per date-symbol-model.
    predictions = predictions.drop_duplicates(subset=["date", "symbol", "model_name"], keep="first")
    return predictions.sort_values(["date", "symbol", "model_name"]).reset_index(drop=True)


def main() -> None:
    feature_path = Path("data/processed/features.parquet")
    features = pd.read_parquet(feature_path)
    predictions = run_walkforward(features)
    predictions.to_parquet("data/processed/predictions.parquet", index=False)


if __name__ == "__main__":
    main()
