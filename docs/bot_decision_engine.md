# Bot Decision Engine

## Purpose

Define the research-safe decision layer that converts validated model outputs into explicit signals without placing live orders.

The engine is deliberately separated from model training, backtesting, and execution. It may be used for paper trading only after the relevant out-of-sample gates pass.

## Pipeline

```text
market data + macro + events
            |
            v
       feature snapshot
            |
            v
     validated model(s)
            |
            v
   probability / forecast
            |
            v
      decision engine
       /     |      \
   ENTER   HOLD    EXIT
      \      |      /
       risk / exposure filter
              |
              v
       paper-trade signal
```

## Decision contract

Every decision must contain:

- timestamp and market/session identifier;
- model/version and feature-schema version;
- prediction probability or forecast;
- calibrated confidence, when calibration is available;
- action: `LONG`, `SHORT`, `FLAT`, or `NO_TRADE`;
- reason codes explaining the decision;
- risk budget and proposed exposure;
- transaction-cost and slippage assumptions;
- data-quality status;
- provenance/experiment identifier.

## Safety gates

The engine returns `NO_TRADE` when any mandatory condition fails:

1. required data are missing or stale;
2. the model/version is not approved for the current experiment;
3. calibration or validation status is not available;
4. confidence is below the configured threshold;
5. estimated edge does not exceed estimated costs plus safety margin;
6. risk limits would be exceeded;
7. the current regime is outside the validated operating envelope;
8. the lockbox result is not yet released for the strategy;
9. execution mode is not explicitly enabled.

## Signal hierarchy

`NO_TRADE` is the default state.

A directional signal requires both model evidence and economic edge. A high probability alone is insufficient. The decision layer should compare expected advantage with estimated spread, commission, slippage and a configurable safety margin.

Suggested initial research thresholds:

- minimum calibrated confidence: configurable, not hard-coded;
- minimum expected edge after costs: strictly positive;
- maximum portfolio exposure: configurable risk limit;
- maximum turnover: configurable;
- maximum drawdown halt: configurable research control.

These values must be supplied by experiment configuration and never tuned using the final lockbox observations.

## Paper-trading mode

The first operational mode is paper trading. It records hypothetical orders and fills but has no broker/exchange integration and cannot send live orders.

Required paper-trade records:

- signal timestamp;
- requested action and size;
- reference price;
- assumed fill price;
- costs and slippage;
- position before/after;
- realized and unrealized P&L;
- model/provenance metadata.

## Live execution boundary

No live execution should be implemented until A/B/C have passed statistical and financial gates, robustness checks have been completed, and the lockbox is frozen and released. A future execution adapter must be a separate module with an explicit kill switch and independent credentials/configuration.

## Research metrics

The decision engine should report, at minimum:

- signal count and trade frequency;
- hit rate as a secondary metric;
- expected edge;
- gross and net return;
- transaction costs;
- slippage;
- turnover;
- maximum drawdown;
- Sharpe and Sortino where statistically appropriate;
- benchmark comparison;
- confidence intervals and robustness status.

## Implementation order

1. Define a typed JSON decision schema.
2. Implement a pure, deterministic decision function.
3. Add unit tests for every safety gate and boundary condition.
4. Add a paper-trade ledger.
5. Connect only to validated OOS model outputs.
6. Integrate with the research dashboard as read-only output.
7. Add monitoring and drift checks.
8. Keep live execution as a separate future stage.
