# Real Backtest Evaluation Protocol

This document defines how the real historical A/B/C lockbox run is interpreted. It is intentionally separate from model fitting and does not change the frozen experiment.

## 1. Validity gates

A result is considered valid only if the run completes the full chain:

`real staging -> PIT materialization -> A/B/C execution -> untouched chronological lockbox -> financial matrix -> stability -> reports`.

The final lockbox must remain untouched for model/threshold/feature selection. Any post-lockbox tuning invalidates the reported lockbox result and requires a new frozen experiment.

## 2. Information-set rule

Market features are computed from information available at the decision timestamp. Historical events are admitted through `available_at`; `published_at` and `event_time` are provenance fields, not substitutes for availability. The event feature builder therefore cannot use an event before its information cutoff.

## 3. Primary comparisons

The financial evaluation compares the frozen A/B/C experiments on the same out-of-sample index and includes:

- fixed transaction-cost grid: 0, 5 and 10 bps;
- fixed slippage grid: 0 and 5 bps;
- random baseline with fixed seed;
- buy-and-hold benchmark.

No cell in the matrix is selected after inspecting the lockbox to claim a better result.

## 4. Decision framework

Interpret results in this order:

1. **Protocol validity:** no failed audit, missing data, or look-ahead violation.
2. **Predictive performance:** compare out-of-sample classification metrics against the relevant baseline.
3. **Economic performance:** compare returns, volatility, drawdown and risk-adjusted metrics after costs and slippage.
4. **Robustness:** require the conclusion to survive reasonable costs/slippage rather than relying on a single frictionless cell.
5. **Temporal stability:** inspect early, middle and late lockbox periods. A strong aggregate result with one isolated period is not treated as universally robust.
6. **Benchmark superiority:** distinguish statistical improvement from economically useful improvement over buy-and-hold and random controls.

## 5. Result labels

### Strong evidence
The experiment beats the appropriate baselines out of sample, remains economically positive after realistic costs/slippage, and shows consistent temporal behavior.

### Conditional evidence
The experiment shows an advantage, but it depends materially on costs, a particular period, or another explicitly reported condition.

### No convincing edge
The model does not reliably beat the baselines after costs, or its advantage is not stable across time.

### Invalid / inconclusive
The protocol or data audit fails, the lockbox is contaminated, required data are unavailable, or the execution did not complete the complete evaluation chain.

## 6. Statistical caution

Confidence intervals and significance-style summaries are descriptive evidence, not proof of future profitability. A result must not be promoted to a production strategy solely because one metric or one cost cell is favorable.

## 7. IA phase boundary

The current real backtest remains the non-IA benchmark. An eventual IA layer must be evaluated as a separate experiment against this frozen baseline, using a new protocol version and a fresh untouched evaluation period where possible. This prevents improvements from being attributed to the IA layer when they actually come from changing the underlying experimental rules.
