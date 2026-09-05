# Research lifecycle and safety gate

The project follows this mandatory order:

```text
DATA
  -> TEMPORAL AUDIT
  -> POINT-IN-TIME INFORMATION SET
  -> A/B/C SHARED PROTOCOL
  -> LOCKBOX FREEZE
  -> OOS PREDICTIONS
  -> FINANCIAL BACKTEST
  -> FINAL REPORT
```

## Data

Every source must preserve provenance, coverage, retrieval identity and a reproducible hash where practical. Historical macro data must retain vintage availability. Events must preserve event time separately from publication/availability time.

## Temporal audit

The audit checks timezone-aware timestamps, chronology, future availability, weekend decisions, duplicate identifiers and invalid numeric values. Market sessions must be interpreted using the exchange calendar rather than assuming every UTC date is tradable.

## Information set

For every historical decision timestamp, reconstruct only information available by that timestamp. Market observations, macro vintages and event records are admitted independently. No future-value substitution is permitted.

## A/B/C

A, B and C share one immutable `ABCProtocol`, one fold sequence, one observation index, one target/horizon/purge configuration and one lockbox. Their only intended difference is feature information: technical, technical+macro, and technical+macro+events.

## Lockbox

The final OOS lockbox is not a tuning set. Protocol and configuration identities must be frozen before evaluation. Any material protocol change invalidates the lockbox identity.

## Financial evaluation

The fixed matrix evaluates 0/5/10 bps transaction costs and 0/5 bps slippage, with common OOS observations and benchmark comparisons. Report risk, drawdown, turnover and temporal stability; do not select a winner using the final lockbox.

## D / AI

D is strictly C + AI overlay. AI must use the frozen C lockbox and identical observations. D measures incremental value only and cannot replace the quantitative C predictor.

## Final report

A final report is valid only after the preceding gates succeed. It must document predictive performance, calibration, uncertainty, temporal stability, financial robustness, benchmarks, leakage checks and limitations.
