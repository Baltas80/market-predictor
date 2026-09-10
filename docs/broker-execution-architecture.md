# Broker execution architecture

Market Predictor is research software first. Broker integration is an execution boundary and must not alter the historical lockbox protocol.

## Design

```text
signal/model -> risk gate -> execution policy -> BrokerAdapter -> broker API
                       |                 |
                       +-> audit log    +-> idempotency/client order id
```

The model never receives broker credentials and must not be able to change global risk limits.

## Modes

1. **Simulation** — local deterministic fills for tests and research.
2. **Paper trading** — broker sandbox when the selected provider supports one.
3. **Live, constrained** — explicit account-level limits and a kill switch.

Live trading is intentionally not implemented by this interface alone.

## Required controls before live routing

- Maximum order notional.
- Maximum position exposure.
- Maximum daily loss.
- Maximum turnover/rate of orders.
- Allowed symbols/asset classes.
- Duplicate-order/idempotency protection.
- Broker-side order status reconciliation.
- Explicit kill switch that blocks new orders.
- Full audit trail containing decision, risk result, order request and broker acknowledgement.

## Research isolation

Broker execution must never feed future fills, account state, or post-decision information back into historical features or lockbox evaluation. The sequence remains:

`real data -> point-in-time -> A/B/C -> final OOS lockbox -> robustness -> financial validation -> D/AI -> production`

A future live adapter should be added in a separate implementation module and tested against a broker sandbox before any production credentials are accepted.
