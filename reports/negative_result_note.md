# Regime-Conditioned ETF Direction Forecasting: A Preregistered Walk-Forward Study (Negative Result)

## Hypothesis

Test whether simple daily features from liquid ETFs contain state-dependent information about next-day returns, and whether conditioning on volatility regimes improves out-of-sample forecast quality enough to survive realistic trading costs.

## Methods

- Universe: `SPY`, `QQQ`, `IWM`, `TLT`, `HYG`, `GLD`
- Features: `ret_5d`, `ret_20d`, `rv_20d`, `vol_ratio`, `spy_ret_20d`, `spy_rv_20d`
- Walk-forward: 3 years train, 6 months test, 3 months step
- Forecast targets: `y_up = 1[r_{t+1} > 0]`
- Models tested:
  - `logit_plain`
  - `logit_regime` in Week 1 only
  - baselines: `base_rate`, `mom_sign_20d`
- Portfolio mapping:
  - initial mapping used `2p - 1`
  - final evaluation used drift-neutral centering on the train base rate: `2 * (p_up - p_train)`
  - weights were volatility-scaled and capped
- Costs: 5 bps and 10 bps
- Economic baselines:
  - `buy_hold_equal_weight`
  - `equal_weight_daily`
  - `equal_weight_monthly`
  - `vol_target_only`

## Preregistered Priors And Kill Rules

Week 1 prior:

> I expect regime conditioning to improve out-of-sample log loss and Brier score by roughly 1-2% versus the plain logistic model, with most of the benefit concentrated in high-volatility periods, while any portfolio improvement will be modest and may not survive 10 bps transaction costs.

Kill rule for regime conditioning:

- drop it if it does not improve forecast quality versus `logit_plain`
- drop it if the improvement is not stable across folds
- drop it if the effect is not stronger in high-volatility periods

## Results

### Week 1: Regime Conditioning Failed

- `logit_regime` underperformed `logit_plain` on both forecast metrics:
  - log loss: `0.6951` vs `0.6927`
  - Brier: `0.2508` vs `0.2497`
- Fold win rate was `31.88%` across `69` walk-forward folds.
- High-volatility performance was worse, not better:
  - high-vol log loss: `0.7001` vs `0.6973`
  - high-vol Brier: `0.2530` vs `0.2519`

Decision: remove regime conditioning from the research path.

### Drift-Neutral Recheck Eliminated The Apparent Edge

The initial portfolio mapping around `0.5` was monetizing market drift. After centering each fold on the train base rate, the economic edge disappeared.

Week 2 out-of-sample forecast metrics:

- `base_rate`: log loss `0.6909`, Brier `0.2489`
- `logit_plain`: log loss `0.6927`, Brier `0.2497`

Week 2 portfolio metrics at 5 bps:

- `vol_target_only`: Sharpe `1.04`
- `equal_weight_monthly`: Sharpe `1.03`
- `equal_weight_daily`: Sharpe `1.03`
- `buy_hold_equal_weight`: Sharpe `0.93`
- `logit_plain`: Sharpe `0.03`

`logit_plain` failed both the forecast comparison and the economic baseline comparison.

## Why It Failed / What We Learned

1. Regime conditioning did not add information. It degraded forecast quality, failed fold stability, and showed the wrong sign in the high-volatility regime it was supposed to help.
2. The earlier positive Sharpe was mostly a portfolio-construction artifact. Mapping around `0.5` retained a long bias in a market with positive drift.
3. Once the signal was centered on the train base rate, the forecast did not produce tradable edge.
4. Simple allocation baselines dominated the model. Risk scaling and diversified passive exposure explained more than the daily forecast did.

## Credibility Checks

- Shuffled-target forecasts remained close to `base_rate` and did not produce a meaningful edge.
- Target weights stamped at date `t` are applied to `fwd_ret_1d` from `t` to `t+1`.
- Turnover is measured against post-return, pre-trade weights, so passive drift is not counted as trading.
- `equal_weight_daily` average turnover at 5 bps was `0.61%` per day versus `0.15%` for `equal_weight_monthly`, which explains why the two equal-weight baselines have similar net Sharpe despite different rebalance frequencies.

## Conclusion

Under this setup, daily ETF direction forecasting with these simple features does not add incremental value over a base-rate forecast and does not survive comparison against straightforward economic baselines after costs.

This is a negative result, but it is a valid research outcome:

- the hypothesis was preregistered
- the signal was stress-tested against leakage and cost artifacts
- complexity was removed when it did not pay

## Next Experiment

Change one dimension only: keep the same features, costs, and baselines, but move to a lower-frequency decision rule.

Recommended next test:

- forecast weekly returns or keep the daily forecast and rebalance weekly

That isolates whether the failure came from daily noise and turnover rather than from the feature set alone.
