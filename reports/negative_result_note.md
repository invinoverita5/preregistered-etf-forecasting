# Regime-Conditioned ETF Direction Forecasting: A Preregistered Walk-Forward Study (Negative Result)

## Hypothesis

Test whether simple features from liquid ETFs contain state-dependent information about short-horizon returns, and whether that information survives realistic costs once translated into a tradable portfolio.

## Methods

- Universe: `SPY`, `QQQ`, `IWM`, `TLT`, `HYG`, `GLD`
- Features: `ret_5d`, `ret_20d`, `rv_20d`, `vol_ratio`, `spy_ret_20d`, `spy_rv_20d`
- Walk-forward: 3 years train, 6 months test, 3 months step
- Forecast targets:
  - Week 1-3: `y_up = 1[r_{t+1} > 0]`
  - Week 4: `y_up = 1[r_{t+5} > 0]`
  - Week 5: `y_up = 1[r_{t+10} > 0]`
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

### Week 2: Drift-Neutral Recheck Eliminated The Apparent Edge

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

### Week 3: Weekly Rebalance Reduced Turnover But Did Not Rescue The Signal

Changing only the forecast-driven rebalance schedule from daily to weekly reduced turnover but did not improve the economics:

- `logit_plain` 5 bps Sharpe moved from `0.03` to `-0.04`
- average turnover fell from `60.55%` to `29.36%`
- forecast quality remained worse than `base_rate`

This isolated the problem further: the failure was not simply daily trading costs.

### Week 4: 5-Day Target Horizon Failed

Changing only the model label from 1 day to 5 trading days, while keeping weekly rebalance and the same baselines, still failed:

- `base_rate`: log loss `0.6842`, Brier `0.2455`
- `logit_plain`: log loss `0.6892`, Brier `0.2475`
- `logit_plain` 5 bps Sharpe: `-0.03`

The 5-day label did not recover forecast edge, and it increased turnover versus the prior weekly-rebalance daily-target run.

### Week 5: 10-Day Target Horizon Also Failed

The final one-dimensional extension moved the label to 10 trading days:

- `base_rate`: log loss `0.6789`, Brier `0.2429`
- `logit_plain`: log loss `0.6845`, Brier `0.2449`
- `logit_plain` 5 bps Sharpe: `0.04`
- `vol_target_only` 5 bps Sharpe: `1.04`

The 10-day label slightly improved net Sharpe relative to the 5-day run, but it still failed the primary test:

- forecast quality remained worse than `base_rate`
- economic performance remained far behind the passive and risk-scaled baselines

## Why It Failed / What We Learned

1. Regime conditioning did not add information. It degraded forecast quality, failed fold stability, and showed the wrong sign in the high-volatility regime it was supposed to help.
2. The earlier positive Sharpe was mostly a portfolio-construction artifact. Mapping around `0.5` retained a long bias in a market with positive drift.
3. Once the signal was centered on the train base rate, the forecast did not produce tradable edge.
4. Lowering trading frequency did not rescue the signal family. Weekly rebalance reduced turnover, but not enough to create edge.
5. Extending the target horizon to 5 and 10 trading days also failed. The 10-day label was less bad economically than the 5-day label, but still not good enough to beat a trivial base-rate forecast or simple baseline portfolios.
6. Simple allocation baselines dominated throughout. Risk scaling and diversified passive exposure explained far more than the forecast did.

## Credibility Checks

- Shuffled-target forecasts remained close to `base_rate` and did not produce a meaningful edge.
- Target weights stamped at date `t` are applied to `fwd_ret_1d` from `t` to `t+1`.
- Turnover is measured against post-return, pre-trade weights, so passive drift is not counted as trading.
- `equal_weight_daily` average turnover at 5 bps was `0.61%` per day versus `0.15%` for `equal_weight_monthly`, which explains why the two equal-weight baselines have similar net Sharpe despite different rebalance frequencies.

## Conclusion

Under this setup, this signal family does not add incremental value over a base-rate forecast and does not survive comparison against straightforward economic baselines after costs.

This is a negative result, but it is a valid research outcome:

- the hypothesis was preregistered
- the signal was stress-tested against leakage and cost artifacts
- complexity was removed when it did not pay

## Final Disposition

Stop exploring this signal family.

The final sequence tested:

- regime conditioning
- drift-neutral daily forecasting
- weekly rebalance
- 5-day target horizon
- 10-day target horizon

None of these variants beat the forecast base rate or the simple economic baselines after costs. The most defensible conclusion is that this line of research, as currently specified, should be closed rather than tuned further.
