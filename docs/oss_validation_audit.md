# OSS validation audit

## Purpose

Evaluate external validation libraries without changing the project's point-in-time (PIT) dataset, historical lockbox, or final OOS parameters.

## Candidates

- `purgedcv`: independent implementation of purge, embargo, chronological walk-forward and CPCV. Compare its split boundaries against `ABCProtocol.common_walk_forward_folds` and `ABCExecutionPlan`.
- `oos-lab`: independent statistical validation layer for Sharpe, PSR, Deflated Sharpe, PBO/CSCV and multiple-testing haircuts. Compare only after the final financial return series is available.

## Acceptance rules

1. The project's PIT and lockbox remain authoritative.
2. No external package may modify lockbox dates, purge horizon, target definition, transaction costs, slippage, or OOS observations.
3. For an equivalent synthetic dataset, external and internal chronological splitters must produce identical train/test boundaries where the methodologies are declared equivalent.
4. Any statistical discrepancy must be reported rather than silently replaced.
5. Dependencies are not added to the production package until equivalence tests and CI pass.
6. No historical workflow trigger is created from this audit branch.

## Current internal contract

`ABCProtocol` freezes lockbox dates, purge gap, horizon, costs, observation count, dataset/code/protocol identity and fold identity. `ABCExecutionPlan` additionally validates chronological unique observations, purge separation and exact OOS prediction-index identity.

## Next implementation steps

- Add isolated synthetic equivalence tests for `purgedcv` against the internal fold generator.
- Add an optional validation adapter for `oos-lab` that consumes already-computed OOS returns only.
- Keep both packages optional until the comparison is complete.
- If equivalence is demonstrated, document the external library as an independent audit rather than replacing the project's authoritative lockbox implementation.
