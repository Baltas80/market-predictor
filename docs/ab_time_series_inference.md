# A/B time-series inference protocol

The A/B predictive comparison is paired: both models are evaluated on the exact same chronological observations and realized labels. Because adjacent market observations are dependent, ordinary IID confidence intervals are not accepted as the primary inference control.

## Bootstrap

`paired_block_bootstrap_difference()` uses a circular moving-block bootstrap. The same resampled timestamps are applied to A and B, preserving the paired design and short-range temporal dependence. The reported effect is `B - A`.

The production analysis uses a pre-declared sensitivity grid of block lengths rather than choosing one block length after seeing the result. For the current 5-session prediction horizon, block lengths 5, 20 and 60 are used as a robustness grid.

## Permutation

`paired_block_swap_test()` creates a null distribution by independently swapping A/B predictions at the level of contiguous time blocks. This preserves local dependence while testing paired exchangeability. The p-value is two-sided and uses the finite-sample +1 correction.

## Multiple comparisons

`benjamini_hochberg()` is available for false-discovery control when multiple related hypotheses are reported. Adjusted p-values must be reported alongside raw p-values; they are not used to select a preferred model.

## Scientific boundary

These methods operate only after the upstream chronological OOS predictions have been generated. They do not tune features, models, thresholds, costs, slippage, or lockbox periods. A statistical result is uncertainty quantification, not proof of economic value or causal predictive power.
