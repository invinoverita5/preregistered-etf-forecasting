from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import forecast_metrics, portfolio_summary_table

SUMMARY_NOTE = (
    "Week 2 baseline comparison: regime conditioning was removed after failing the preregistered Week 1 criteria. "
    "This run asks whether the drift-neutral plain logistic forecast adds value over simple economic baselines."
)

FORECAST_ORDER = [
    "base_rate",
    "mom_sign_20d",
    "logit_plain",
    "logit_plain_shuffled",
]
MAIN_PORTFOLIO_ORDER = [
    "buy_hold_equal_weight",
    "equal_weight_daily",
    "equal_weight_monthly",
    "vol_target_only",
    "mom_sign_20d",
    "logit_plain",
]
APPENDIX_ORDER = [f"buy_hold_{symbol}" for symbol in ["SPY", "QQQ", "IWM", "TLT", "HYG", "GLD"]]
EXECUTION_ORDER = [
    "buy_hold_equal_weight",
    "equal_weight_daily",
    "equal_weight_monthly",
    "vol_target_only",
    "logit_plain",
]


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def _ordered_models(df: pd.DataFrame, order: list[str], model_col: str = "model_name") -> pd.DataFrame:
    order_map = {name: idx for idx, name in enumerate(order)}
    sort_cols = ["_order", model_col]
    if "cost_bps" in df.columns:
        sort_cols.append("cost_bps")
    return df.assign(_order=df[model_col].map(order_map).fillna(len(order_map))).sort_values(sort_cols).drop(
        columns="_order"
    )


def _relative_improvement(baseline_value: float, model_value: float) -> float:
    if pd.isna(baseline_value) or baseline_value == 0:
        return np.nan
    return float((baseline_value - model_value) / baseline_value)


def _get_model_row(df: pd.DataFrame, model_name: str) -> pd.Series | None:
    matched = df.loc[df["model_name"] == model_name]
    if matched.empty:
        return None
    return matched.iloc[0]


def _fold_base_rate_table(predictions: pd.DataFrame) -> pd.DataFrame:
    keep = predictions[predictions["model_name"] == "base_rate"].copy()
    if keep.empty:
        return pd.DataFrame()

    overall = (
        keep.groupby("fold_id", as_index=False)
        .agg(
            train_start=("train_start", "first"),
            train_end=("train_end", "first"),
            test_start=("test_start", "first"),
            test_end=("test_end", "first"),
            train_base_rate=("train_base_rate", "first"),
        )
        .sort_values("fold_id")
    )

    symbol_rates = (
        keep.pivot_table(
            index="fold_id",
            columns="symbol",
            values="train_symbol_base_rate",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    return overall.merge(symbol_rates, on="fold_id", how="left")


def _portfolio_lookup(portfolio_metrics_df: pd.DataFrame, model_name: str, cost_bps: float) -> pd.Series | None:
    matched = portfolio_metrics_df[
        (portfolio_metrics_df["model_name"] == model_name) & (portfolio_metrics_df["cost_bps"] == cost_bps)
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def _execution_sanity_table(portfolio_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_metrics_df.empty:
        return pd.DataFrame()

    keep = portfolio_metrics_df[
        (portfolio_metrics_df["cost_bps"] == 5.0) & (portfolio_metrics_df["model_name"].isin(EXECUTION_ORDER))
    ].copy()
    if keep.empty:
        return pd.DataFrame()

    cols = ["model_name", "ann_return", "sharpe", "avg_turnover"]
    return _ordered_models(keep.loc[:, cols], EXECUTION_ORDER)


def _sanity_check(forecast_tbl: pd.DataFrame) -> tuple[bool, list[str]]:
    notes: list[str] = []
    base_rate_row = _get_model_row(forecast_tbl, "base_rate")
    shuffled_row = _get_model_row(forecast_tbl, "logit_plain_shuffled")
    suspicious = False

    if base_rate_row is not None and shuffled_row is not None:
        ll_lift = _relative_improvement(base_rate_row["log_loss"], shuffled_row["log_loss"])
        br_lift = _relative_improvement(base_rate_row["brier_score"], shuffled_row["brier_score"])
        if (not pd.isna(ll_lift) and ll_lift > 0.005) or (not pd.isna(br_lift) and br_lift > 0.005):
            suspicious = True
            notes.append("logit_plain_shuffled improved forecast metrics versus base_rate beyond the sanity tolerance.")

    if not notes:
        notes.append("Shuffled-target forecasts remained close to base_rate and did not show a meaningful edge.")
    return (not suspicious), notes


def main() -> None:
    prediction_path = Path("data/processed/predictions.parquet")
    portfolio_path = Path("data/processed/portfolio_daily.parquet")
    predictions = pd.read_parquet(prediction_path)
    portfolio = pd.read_parquet(portfolio_path) if portfolio_path.exists() else pd.DataFrame()

    forecast_tbl = _ordered_models(forecast_metrics(predictions), FORECAST_ORDER)
    fold_base_rates = _fold_base_rate_table(predictions)
    portfolio_tbl = portfolio_summary_table(portfolio) if not portfolio.empty else pd.DataFrame()
    execution_tbl = _execution_sanity_table(portfolio_tbl)

    main_tbl = (
        _ordered_models(
            portfolio_tbl[portfolio_tbl["model_name"].isin(MAIN_PORTFOLIO_ORDER)].copy(),
            MAIN_PORTFOLIO_ORDER,
        )
        if not portfolio_tbl.empty
        else pd.DataFrame()
    )
    appendix_tbl = (
        _ordered_models(
            portfolio_tbl[portfolio_tbl["model_name"].isin(APPENDIX_ORDER)].copy(),
            APPENDIX_ORDER,
        )
        if not portfolio_tbl.empty
        else pd.DataFrame()
    )

    base_row = _get_model_row(forecast_tbl, "base_rate")
    mom_row = _get_model_row(forecast_tbl, "mom_sign_20d")
    plain_row = _get_model_row(forecast_tbl, "logit_plain")

    plain_vs_base_ll = (
        _relative_improvement(base_row["log_loss"], plain_row["log_loss"]) if base_row is not None and plain_row is not None else np.nan
    )
    plain_vs_base_br = (
        _relative_improvement(base_row["brier_score"], plain_row["brier_score"])
        if base_row is not None and plain_row is not None
        else np.nan
    )
    plain_vs_mom_ll = (
        _relative_improvement(mom_row["log_loss"], plain_row["log_loss"]) if mom_row is not None and plain_row is not None else np.nan
    )
    plain_vs_mom_br = (
        _relative_improvement(mom_row["brier_score"], plain_row["brier_score"])
        if mom_row is not None and plain_row is not None
        else np.nan
    )

    plain_port_5 = _portfolio_lookup(portfolio_tbl, "logit_plain", 5.0) if not portfolio_tbl.empty else None
    ew_daily_5 = _portfolio_lookup(portfolio_tbl, "equal_weight_daily", 5.0) if not portfolio_tbl.empty else None
    ew_monthly_5 = _portfolio_lookup(portfolio_tbl, "equal_weight_monthly", 5.0) if not portfolio_tbl.empty else None
    best_baseline_5 = None
    if not main_tbl.empty:
        baseline_rows = main_tbl[(main_tbl["cost_bps"] == 5.0) & (main_tbl["model_name"] != "logit_plain")].copy()
        if not baseline_rows.empty:
            best_baseline_5 = baseline_rows.sort_values("sharpe", ascending=False).iloc[0]

    sanity_pass, sanity_notes = _sanity_check(forecast_tbl)

    forecast_has_incremental_value = bool(
        not pd.isna(plain_vs_base_ll)
        and not pd.isna(plain_vs_base_br)
        and plain_vs_base_ll > 0.0
        and plain_vs_base_br >= 0.0
    )
    forecast_beats_mom = bool(
        not pd.isna(plain_vs_mom_ll)
        and not pd.isna(plain_vs_mom_br)
        and plain_vs_mom_ll > 0.0
        and plain_vs_mom_br >= 0.0
    )
    baseline_gap = (
        float(plain_port_5["sharpe"]) - float(best_baseline_5["sharpe"])
        if plain_port_5 is not None and best_baseline_5 is not None
        else np.nan
    )
    economic_beats_baselines = bool(not pd.isna(baseline_gap) and baseline_gap > 0.0)
    ew_turnover_gap = (
        float(ew_daily_5["avg_turnover"]) - float(ew_monthly_5["avg_turnover"])
        if ew_daily_5 is not None and ew_monthly_5 is not None
        else np.nan
    )
    ew_extra_cost_bps = (
        ew_turnover_gap * (5.0 / 10000.0) * 252.0 * 10000.0 if not pd.isna(ew_turnover_gap) else np.nan
    )

    if not sanity_pass:
        decision = "INVALIDATE results and audit the data path."
    elif forecast_has_incremental_value and economic_beats_baselines:
        decision = "Keep the plain probabilistic forecast alive; it adds value beyond the simple baselines."
    elif forecast_has_incremental_value and not economic_beats_baselines:
        decision = "Forecast quality may contain weak information, but it does not survive the economic baseline comparison."
    else:
        decision = "Reject the plain probabilistic forecast as a tradable signal under the current setup."

    out: list[str] = []
    out.append("# Week 2 Baseline Comparison")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"> {SUMMARY_NOTE}")
    out.append("")
    out.append("## Forecast Metrics (OOS Aggregate)")
    out.append("")
    out.append(_markdown_table(forecast_tbl))
    out.append("")
    out.append("## Forecast Takeaways")
    out.append("")
    out.append(f"- `logit_plain` vs `base_rate`: log loss lift **{plain_vs_base_ll:.2%}**, Brier lift **{plain_vs_base_br:.2%}**.")
    out.append(f"- `logit_plain` vs `mom_sign_20d`: log loss lift **{plain_vs_mom_ll:.2%}**, Brier lift **{plain_vs_mom_br:.2%}**.")
    out.append("")
    out.append("## Fold Base Rates")
    out.append("")
    out.append(_markdown_table(fold_base_rates))
    out.append("")
    out.append("## Main Portfolio Baselines")
    out.append("")
    out.append(_markdown_table(main_tbl))
    out.append("")
    out.append("## Buy-and-Hold Appendix")
    out.append("")
    out.append(_markdown_table(appendix_tbl))
    out.append("")
    out.append("## Execution Sanity")
    out.append("")
    out.append("- Timing convention: target weights stamped at date `t` are applied to `fwd_ret_1d` from `t` to `t+1`.")
    out.append(
        "- Turnover convention: trading is measured against post-return, pre-trade weights, so passive drift is not charged as turnover."
    )
    if not pd.isna(ew_turnover_gap) and not pd.isna(ew_extra_cost_bps):
        out.append(
            f"- `equal_weight_daily` average turnover was **{float(ew_daily_5['avg_turnover']):.2%}** per day versus "
            f"**{float(ew_monthly_5['avg_turnover']):.2%}** for `equal_weight_monthly`; at 5 bps that is roughly "
            f"**{ew_extra_cost_bps:.1f} bps/year** of extra cost."
        )
    out.append("")
    out.append(_markdown_table(execution_tbl))
    out.append("")
    out.append("## Sanity Check")
    out.append("")
    for note in sanity_notes:
        out.append(f"- {note}")
    out.append("")
    out.append("## Decision")
    out.append("")
    if best_baseline_5 is not None and plain_port_5 is not None:
        out.append(
            f"- Best 5 bps baseline: **{best_baseline_5['model_name']}** with Sharpe **{float(best_baseline_5['sharpe']):.2f}**."
        )
        out.append(f"- `logit_plain` 5 bps Sharpe: **{float(plain_port_5['sharpe']):.2f}**.")
        out.append(f"- Sharpe gap vs best baseline: **{baseline_gap:.2f}**.")
    out.append(f"- `logit_plain` beats `mom_sign_20d` on forecast metrics: **{forecast_beats_mom}**.")
    out.append(f"- `logit_plain` beats `base_rate` on forecast metrics: **{forecast_has_incremental_value}**.")
    out.append(f"- `logit_plain` beats the economic baselines at 5 bps: **{economic_beats_baselines}**.")
    out.append(f"- Sanity check pass: **{sanity_pass}**.")
    out.append("")
    out.append(f"**Week-2 decision:** {decision}")

    output_path = Path("reports/week2_baselines.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
