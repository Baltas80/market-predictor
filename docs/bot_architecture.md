# Paper Bot Architecture

## Components

1. **Data adapter** — consumes only timestamped, point-in-time datasets.
2. **Feature snapshot** — produces the exact feature schema used by the approved model.
3. **Model adapter** — loads a versioned A/B/C model and emits predictions plus provenance.
4. **Decision engine** — converts predictions into LONG/SHORT/FLAT/NO_TRADE using frozen configuration.
5. **Risk layer** — enforces exposure, turnover, drawdown and data-quality limits.
6. **Paper broker** — simulates orders/fills and costs; it has no live credentials.
7. **Ledger** — persists every signal, simulated fill and state transition.
8. **Monitor** — detects stale data, model drift, abnormal turnover and risk-limit breaches.
9. **Dashboard adapter** — exposes read-only research and paper-trading state.

## Hard separation

```text
TRAINING / RESEARCH  --->  APPROVED MODEL  --->  PAPER BOT
       |                         |                  |
       |                         |                  +--> NO LIVE ORDERS
       +--> lockbox frozen -----+                  |
                                                  ledger
                                                    |
                                                dashboard
```

The paper bot must never modify training data, lockbox data, model parameters, experiment configuration or historical results.

## State machine

```text
NO_TRADE
   |
   v
DATA_VALID --> PREDICTED --> RISK_CHECK --> SIGNALLED --> PAPER_FILLED
   |              |             |              |
   +--------------+-------------+--------------+
                  |
                  v
               NO_TRADE
```

Any failed validation returns to `NO_TRADE` and records a reason code.

## Future live boundary

A live broker adapter is intentionally outside this architecture for the current phase. If introduced later, it must be isolated from the research engine, require explicit operator enablement, enforce independent risk controls, and include a kill switch.
