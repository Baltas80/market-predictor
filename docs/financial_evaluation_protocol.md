# Financial evaluation protocol

The financial evaluation consumes frozen OOS predictions. It does not retrain models, select thresholds, or modify the lockbox.

## Fixed grid

- Transaction costs: **0, 5, 10 bps**.
- Slippage: **0, 5 bps**.
- Probability threshold: declared before final OOS evaluation.
- Annualization: 252 daily periods unless the dataset declares another frequency.

The matrix therefore contains 6 execution scenarios per experiment, plus the deterministic random baseline: **24 rows for A/B/C/random**.

## Required outputs

For each scenario record total return, CAGR, maximum drawdown, annualized volatility, Sharpe, Sortino, downside deviation, hit rate, average position and turnover. Report buy-and-hold total return, drawdown, Sharpe and Sortino on the same OOS dates.

## Stability

After the main evaluation, use pre-declared chronological periods to report return, drawdown, Sharpe, Sortino and turnover. Periods are descriptive diagnostics, not a mechanism for selecting a final-OOS configuration.

## Baselines

Buy-and-hold is the primary model-free benchmark. A fixed-seed random probability signal is a second reference. Neither baseline is a tuning candidate.
