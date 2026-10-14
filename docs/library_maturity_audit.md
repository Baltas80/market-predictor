# Mature-library adoption policy

## Objective

Reduce maintenance risk by using established scientific Python libraries for generic numerical, statistical and model-building work. Keep the project's research-specific controls explicit and auditable. A popular dependency is not, by itself, evidence that a research result is valid.

## Current dependency baseline

The project already declares NumPy, pandas, scikit-learn and requests in `pyproject.toml`. Prefer these existing dependencies where they fit; do not add a second library that duplicates their core role without a measured benefit.

- **NumPy / pandas:** array and time-indexed tabular operations.
- **scikit-learn:** estimators, probability metrics, calibration and general model evaluation.
- **requests:** HTTP transport for source adapters; source-specific parsing and provenance remain project responsibilities.
- **pytest:** automated regression tests.

## Component decisions

| Component | Policy | Reason / acceptance condition |
| --- | --- | --- |
| Generic estimators and metrics | Prefer scikit-learn | Mature, documented APIs; preserve exact target, feature, threshold and metric definitions. |
| Walk-forward fold generation | Keep the research-specific wrapper | Generic splitters do not automatically enforce the project's label-horizon purge and shared A/B/C fold identity. A standard splitter may be used inside the wrapper only if parity tests prove identical boundaries. |
| Point-in-time availability | Keep explicit project controls | Must encode source-specific publication/availability semantics, not merely chronological row order. |
| A/B/C protocol and lockbox | Do not replace casually | Immutable protocol, common index/fold hashes and fail-closed guards are part of the scientific design. |
| Financial backtesting | Evaluate an external engine in a separate parity experiment | Do not adopt Backtrader, vectorbt or another engine until order timing, price assumptions, costs, slippage, turnover and outputs are shown equivalent to the current research specification. |
| Market calendars | Consider a dedicated calendar library only after coverage audit | Validate historical exchange sessions, exceptional closures, early closes, timezone/DST handling and versioning against an authoritative reference. |
| Data ingestion | Use official provider interfaces where available | Every adapter must preserve raw artifacts, retrieval metadata, coverage, hashes and availability fields. |
| LLM/event extraction | Keep outside the core dependency baseline until phase D | Must not change C's frozen data, protocol or lockbox. |

## Required acceptance process for a replacement

1. Pin or bound supported dependency versions and check the project's Python >=3.11 compatibility.
2. Review project health, release history, license, transitive dependencies and security posture.
3. Write deterministic parity tests against the current implementation using synthetic edge cases and representative fixed fixtures.
4. Test boundaries: empty inputs, one-row inputs, gaps, duplicated timestamps, timezone-aware timestamps, DST transitions, missing values and label horizons.
5. Compare outputs and indexes, not only aggregate scores.
6. Run the complete test suite and CI on the proposed branch.
7. Record the dependency/version, rationale, known differences and rollback path.
8. Do not use final lockbox outcomes to select a replacement or tune its parameters.

## Research invariants that must remain true

- All data used at a decision time was available by that time under a documented source-specific rule.
- Features do not contain future labels or post-decision information.
- Train/test folds are chronological and the purge is at least the target horizon where labels overlap.
- A, B and C use the same eligible observations, fold sequence, lockbox, costs and decision rules.
- Dataset, index, fold and protocol hashes are reproducible.
- A download or schema check alone never makes a dataset eligible for the final lockbox.

## Ingestion documentation discrepancy

The runbook states that `--execute` is not wired to retrieval, while the staging protocol previously described it as a full historical run. Treat the runbook's fail-closed statement as the safety rule until the actual executor and CI workflow are inspected and the docs are reconciled. Never infer that real downloads occurred merely because a workflow invoked `--execute`.

## Deferred decisions

No additional backtesting, market-calendar or statistical dependency is added by this policy. Add one only in a dedicated change with a concrete need, license/dependency review, parity tests and CI evidence.
