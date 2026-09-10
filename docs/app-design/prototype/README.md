# Market Predictor — UI prototype

This static prototype turns the approved product/UI specification into a concrete visual reference. It is intentionally dependency-free so it can be reviewed directly from the repository and later used as the handoff reference for the native mobile implementation.

## Visual baseline

- Dark-first financial dashboard.
- Graphite surfaces with restrained cyan/green/red semantic accents.
- Large numeric hierarchy and compact market cards.
- Persistent environment state: `RESEARCH`, `PAPER`, `LIVE`.
- AI agents are informative/propositional; the independent Risk Gate remains authoritative.
- Live execution is never the default path.
- Point-in-time timestamps are visible wherever prediction/event data is shown.

## Prototype screens

`index.html` contains four representative states:

1. Dashboard — portfolio, regime, signals, agents and Risk Gate.
2. Market detail — price chart, signal, confidence and point-in-time events.
3. Agents — ORÁCULO, ATLAS, NEWS AI and GUARDIAN with auditable state.
4. Execution review — order details, risk decision and explicit submission control.

The prototype is a visual handoff, not the production trading client. It contains no broker credentials, network calls or order execution.

## Implementation mapping

| Prototype element | Production component |
|---|---|
| Environment badge | Environment state + live confirmation policy |
| Metric cards | Portfolio/account read model |
| Signal card | Prediction/signal domain model |
| Agent cards | Versioned agent decision records |
| Risk Gate | Independent execution risk module |
| Execution timeline | Broker execution/audit events |
| Event markers | Point-in-time event store |
| Bottom navigation | Mobile navigation shell |

## Acceptance criteria for native implementation

- No production screen may imply that a prediction is a guaranteed return.
- Every actionable signal exposes timestamp, model/version and confidence/calibration metadata.
- Every live order exposes the Risk Gate decision before submission.
- Duplicate client order IDs are idempotent.
- `LIVE` requires explicit user action and a visible safety state.
- Direction must never be communicated by colour alone.
- Primary touch targets should remain at least 44dp.
