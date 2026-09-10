# Paper Bot Risk Policy

The bot is research-only. These controls prevent a prediction from becoming an unchecked trading instruction.

## Mandatory pre-trade checks

- Data freshness and completeness pass.
- Model is an explicitly approved version.
- Feature schema matches the model.
- Probability is within [0, 1].
- Confidence is available when required by the experiment.
- Expected edge after estimated costs exceeds the configured margin.
- Proposed exposure is within the frozen risk budget.
- Turnover limit is not exceeded.
- Drawdown stop is not breached.
- The strategy is inside its validated operating envelope.

Failure of any mandatory check produces `NO_TRADE` and one or more deterministic reason codes.

## No silent assumptions

Costs, slippage, thresholds and risk limits must come from versioned configuration. The bot must not infer favorable values from recent performance and must not tune itself from the lockbox.

## Emergency state

Any data-integrity failure, unexpected model version, corrupted ledger state or monitoring failure moves the bot to `NO_TRADE` until explicitly recovered.

## Execution boundary

This policy permits paper orders only. Live order routing is prohibited at this stage.
