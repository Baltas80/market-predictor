# Market Predictor — Agent Constitution

This file is the operating contract for Codex agents and other automated coding runners working in this repository.

## Mission

Build a reproducible research system that tests whether market, macroeconomic, geopolitical and event information provides out-of-sample predictive value for future market movements.

This is research software. Do not optimize the system to produce attractive backtest results.

## Non-negotiable research rules

1. **Point-in-time data only.** A feature may use only information available at the prediction decision time.
2. **No look-ahead bias.** Future labels, revised macro values, future event metadata, or future-derived transformations must never enter features.
3. **Chronological evaluation.** Training and testing must respect time order. Do not use random train/test splitting for temporal model evaluation.
4. **Purge future labels.** If the target uses a future horizon H, enforce the required purge gap between training and test observations.
5. **Untouched final lockbox.** Never use the final OOS lockbox to tune features, thresholds, hyperparameters, event mappings, model choice, prompts, AI models, costs, or any other research decision.
6. **A/B/C comparability.** A = technical, B = technical + macro, C = technical + macro + events. They must use the same final lockbox, purge policy, evaluation dates and financial assumptions.
7. **D is incremental.** D = C + AI. AI must be evaluated as incremental information/feature value, not as an unconstrained price oracle.
8. **Benchmarks are mandatory.** Compare against buy-and-hold and appropriate null/simple baselines before claiming predictive or financial value.
9. **Financial realism.** Backtests must account for execution timing, transaction costs, slippage and turnover where applicable.
10. **Reproducibility.** Record dataset provenance, configuration, code version and deterministic seeds where randomness is used.
11. **No silent data repair.** Invalid data should fail validation or be explicitly transformed with a documented rule and test.
12. **Claims require evidence.** Separate predictive accuracy from financial performance, statistical uncertainty and robustness.

## Agent isolation and Git workflow

- Work in a dedicated branch/worktree per task.
- Never directly rewrite another agent's branch.
- Keep commits small, coherent and reviewable.
- Before opening a PR, rebase or otherwise synchronize with the current target branch as appropriate.
- Do not merge another PR merely because it is green if its base is stale or its changes conflict with newer research safeguards.
- Do not modify the final lockbox data or evaluation configuration as part of exploratory tuning.

## Required task workflow

### 1. Inspect

Before changing code:
- inspect the relevant modules and tests;
- identify the current branch/base and existing safeguards;
- search for existing implementations before creating duplicates.

### 2. Implement

- Prefer the smallest correct change.
- Preserve existing public APIs unless a change is justified and tested.
- Add tests for new behavior and regression tests for fixed bugs.
- Keep data-source availability rules explicit.

### 3. Validate

At minimum:
- run the focused tests for the changed module;
- run the full test suite before PR handoff;
- check for temporal leakage when changing data, features, targets, folds or evaluation;
- report failures instead of masking them.

### 4. Handoff

Every PR/task report should state:
- what changed;
- tests run and results;
- branch/commit;
- known limitations;
- whether the change is safe to merge without affecting the lockbox protocol.

## Parallel workstreams

Independent agents may work concurrently on:

- **Statistics:** bootstrap confidence intervals, period stability, significance and null/permutation tests.
- **Robustness:** threshold/cost/slippage sensitivity, model complexity, horizon sensitivity and overfitting stress tests.
- **Benchmarks:** A/B/C/D against buy-and-hold and deterministic/random baselines.
- **Regimes:** crisis, geopolitical shock, high-volatility and normal-regime evaluation using only causal information.
- **Calibration:** Brier score, calibration curves/ECE and probability reliability.
- **AI:** structured event/news extraction for D = C + AI, model comparison and historical-memory safeguards.
- **Financial:** cost accounting, equity curves, turnover/capacity and execution assumptions.
- **QA/reviewer:** leakage audits, regression tests, API consistency and cross-branch integration review.

Agents must avoid overlapping edits where practical. If dependencies exist, document them and hand off through commits/PRs rather than modifying shared work in place.

## Lockbox protection

The final lockbox is a measurement instrument, not a development dataset.

Agents MUST NOT:
- inspect lockbox outcomes to choose among competing implementations;
- repeatedly rerun lockbox evaluation while tuning;
- change lockbox dates after seeing results;
- alter the lockbox because a model performs poorly;
- use lockbox metrics to select an AI model or prompt.

If an experiment requires tuning, use earlier chronological validation data. The lockbox should be run only under a pre-registered/frozen protocol and then treated as final evidence.

## AI-specific rules

AI outputs must be structured, versioned and auditable. At minimum capture:
- source/event identifier;
- source availability timestamp;
- model identifier/version;
- prompt/schema version;
- confidence and structured event attributes;
- processing timestamp.

AI must not receive future market outcomes or future event information. If historical memory is introduced, every memory item must have a point-in-time availability boundary.

Open-source/local models should be compared on the same corpus, schema and downstream quantitative model. Choose on incremental out-of-sample evidence, not conversational quality.

## What agents must not do

- Do not invent data, results, sources or benchmark numbers.
- Do not weaken tests to make CI pass.
- Do not remove leakage checks because they make an experiment inconvenient.
- Do not claim production readiness from a successful unit-test run.
- Do not treat correlation as proof of causal prediction.
- Do not convert an exploratory result into a trading recommendation.

## Definition of done

A change is ready for integration only when it is implemented, tested, documented where necessary, compatible with the temporal/lockbox contract, and its limitations are explicit.

The research sequence remains:

`real data → point-in-time → A/B/C → final OOS lockbox → statistical robustness → financial validation → D/AI incremental test → production`
