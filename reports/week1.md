# Week 1 Preregistered Prior

**Primary prior**

> I expect regime conditioning to improve out-of-sample log loss and Brier score by roughly **1-2%** versus the plain logistic model, with most of the benefit concentrated in **high-volatility periods**, while any portfolio improvement will be **modest** and may not survive **10 bps transaction costs**.

## Secondary Priors

1. **Momentum baseline strength**  
   `mom_sign_20d` should be materially harder to beat than `base_rate`.
2. **Forecast vs economic gap**  
   `logit_regime` should improve **forecast quality** more than **portfolio Sharpe**.
3. **Crisis overfitting warning**  
   If `logit_regime` wins only in a single crisis-heavy fold, the improvement is not stable enough to keep.

## Interpretation Rule

After Week 1, compare:

- Expected vs observed forecast lift
- Expected vs observed regime concentration
- Expected vs observed cost sensitivity

If results differ materially from priors, document whether the difference reflects:

- Market insight
- Model weakness
- Data leakage
- Random variation

## Week 1 Objective

Determine whether **regime conditioning provides real incremental predictive value** over a plain logistic model and whether any improvement translates into economically meaningful performance after realistic costs.

If not, regime conditioning will be removed from the research path.

## Models Compared

- `logit_plain`
- `logit_regime`

All metrics are:

- Out-of-sample only
- Aggregated across walk-forward test windows

## Decision Criteria

### 1. Forecast Quality (Primary)

**Pass if BOTH are true:**

- Relative improvement >= **1%** in **log loss OR Brier score**
- **No degradation** in the other metric

Relative improvement:

`(metric_plain - metric_regime) / metric_plain >= 1%`

If not satisfied -> **FAIL**

### 2. Fold Stability

Compute per-fold metric difference.

**Pass if:**

- `logit_regime` beats `logit_plain` in **>= 60%** of test folds

If improvement comes from one crisis window -> **FAIL**

### 3. Economic Translation

Using volatility-scaled portfolio:

**Pass if:**

- Net Sharpe improvement survives **5 bps**
- Turnover not materially higher (> +25%)

If performance only appears:

- at 0 bps, or
- with large turnover increase

-> **FAIL**

### 4. Regime Consistency

Check forecast quality **by regime**.

**Pass if:**

- Improvement is **stronger in high-vol regime** than low/medium

If regime model does not perform best in the regime it claims to model -> **FAIL**

### 5. Sanity Check

Run shuffled-target experiment.

If shuffled targets produce:

- Similar log loss/Brier
- Positive Sharpe

-> **Assume leakage**  
-> **Invalidate results**

## Kill Rules

| Condition | Decision |
| --- | --- |
| Fail (1) and (2) | **Kill regime conditioning** |
| Fail (1), (2), and (3) | **Kill trading hypothesis** |
| Pass (1) but fail (3) | Keep research, simplify (forecast signal exists but not tradable) |

## Final Week 1 Statement

> Regime conditioning [improved / did not improve] out-of-sample forecast quality versus the plain logistic model, [consistently / inconsistently] across folds, and [did / did not] survive 5 bps transaction costs.  
> Decision: [Continue / Drop regime conditioning / Reconsider hypothesis].
