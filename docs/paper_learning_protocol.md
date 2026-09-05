# Paper-learning protocol

The paper learner is a candidate-generation stage, not an automatic replacement mechanism.

## Contract

1. Only resolved virtual trades may enter the learning table.
2. Every input feature must have been known at the original decision timestamp.
3. `outcome_time` must be strictly after `decision_time`.
4. Experiences are ordered chronologically and split chronologically; the newest block is validation only.
5. A candidate requires both positive and negative training outcomes and a minimum training sample count.
6. The learner never reads final OOS lockbox outcomes to tune itself.
7. A candidate is an artifact to be independently backtested through the existing chronological A/B/C validation protocol before acceptance.
8. Training a candidate never mutates an existing model, lockbox, protocol, dataset, or source manifest.

## Current candidate model

The initial learner is a deterministic logistic-regression meta-model over decision-time paper features: predicted probability, confidence, position and notional. Its target is whether the resolved virtual trade had a positive realized return.

The newest chronological experiences are held out for validation. Validation metrics are diagnostic; they are not used to tune the final OOS lockbox.

## Promotion path

`paper signals → resolved experiences → chronological candidate training → chronological validation → candidate artifact → full backtest → Research Gate → explicit promotion`

No live trading is part of this path. A candidate cannot become the production/final model merely because its paper-learning score is higher.
