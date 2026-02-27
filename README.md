# Trading — Preregistered ETF Forecasting Study (Negative Result)

## About

This repository is a preregistered, walk-forward study of short-horizon ETF return predictability. It implements a minimal end-to-end research pipeline (data -> features/targets -> out-of-sample forecasts -> portfolio construction with costs -> baseline comparison) designed to be falsifiable and resistant to leakage, data-snooping, and accidental drift in signal-to-position mapping.

The current conclusion is a negative result: under the baseline comparison setup, the plain probabilistic forecast does not add incremental economic value beyond simple baselines, and regime conditioning does not improve forecast quality and was dropped.

---

## Contents

- `src/` - implementation (ingest, features, modeling, backtest, evaluation)
- `config/` - frozen universe and experiment config
- `data/` - raw and processed artifacts (parquet)
- `reports/` - research narrative and frozen baseline report
- `tests/` - leakage, timing, and cost regression tests

---

## Method

**Universe:** pooled portfolio across liquid ETFs (see `config/universe.yaml`).

**Targets:** leakage-safe next-day label and forward return (`t` features, `t+1` target).

**Evaluation:** strict time-ordered walk-forward with out-of-sample predictions only. The final prediction table is deduplicated so the evaluated OOS rows are non-overlapping.

**Portfolio construction:** volatility-scaled positions with turnover-based transaction costs. Signal mapping is drift-neutral (centered on the train base rate per fold) to test incremental forecast edge rather than market drift.

**Baselines:** buy and hold (pooled equal-weight plus per-symbol appendix lines), equal-weight (daily/monthly rebalance), vol-target-only, momentum sign baseline, plus a forecast base-rate benchmark.

---

## Repro (local)

### 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the pipeline

```bash
python -m src.data.ingest
python -m src.features.build
python -m src.modeling.walkforward
python -m src.backtest.engine
python -m src.evaluation.report_week1
```

### 3) Run tests

```bash
pytest -q
```

---

## Outputs

The pipeline writes:

- `data/processed/prices.parquet`
- `data/processed/features.parquet`
- `data/processed/predictions.parquet`
- `data/processed/portfolio_daily.parquet`

Reports:

- `reports/negative_result_note.md` - research narrative, preregistered priors, and kill rules
- `reports/week2_baselines.md` - baseline comparison, execution sanity checks (timing/turnover), and final decision

---

## Results (baseline comparison)

Forecast quality (OOS aggregate):

| Model | Log loss | Brier |
| --- | --- | --- |
| base_rate | 0.6909 | 0.2489 |
| logit_plain | 0.6927 | 0.2497 |

Economic baselines (5 bps turnover cost, pooled portfolio):

| Strategy | Sharpe |
| --- | --- |
| vol_target_only | 1.04 |
| equal_weight_monthly | 1.03 |
| equal_weight_daily | 1.03 |
| buy_hold_equal_weight | 0.93 |
| logit_plain | 0.03 |

Appendix (buy and hold per symbol):

- `buy_hold_SPY`
- `buy_hold_QQQ`
- `buy_hold_IWM`
- `buy_hold_TLT`
- `buy_hold_HYG`
- `buy_hold_GLD`

(Per-symbol numbers and additional diagnostics are reported in `reports/week2_baselines.md`.)

**Conclusion:** Under this setup, the plain probabilistic forecast is not incrementally useful as a tradable signal, and regime conditioning was removed after failing preregistered forecast and stability criteria.

---

## Tests and sanity checks

This repo includes tests to make the backtest difficult to fool:

- target shift correctness (uses `t+1`, not `t`)
- walk-forward split ordering and within-fold non-overlap
- transaction cost behavior increases with turnover
- backtest timing: weights at date `t` are applied to `fwd_ret_1d` on the same row
- turnover is computed on the rebalance into the next target, not from passive drift

See `tests/` for details.

---

## Configuration

- `config/universe.yaml` - ETF universe and date range
- `config/experiment_mvp.yaml` - frozen baseline comparison experiment

---

## Notes on interpretation

- The pipeline originally showed non-trivial performance when mapping probabilities around `0.5`. After drift-neutral centering on the train base rate per fold, the apparent edge disappeared, indicating that the earlier results were driven primarily by upward drift plus the mapping rather than incremental predictive power.
- Simple baseline strategies (equal-weight and vol-target-only) dominate the model under this setup; baseline comparisons are treated as the main economic truth checks.

---

## Next steps (if continuing research)

Change one dimension only while keeping the same walk-forward protocol and baselines:

- weekly rebalance (trade less often), or
- weekly target horizon (predict 5d/10d returns)

---

## License

Add a license file if you plan to publish beyond personal or portfolio use.
