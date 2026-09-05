# Market Predictor — Copilot Instructions

## Mission
This repository researches whether market data, macroeconomic information, and geopolitical/events information improve prediction of future market movements. Code changes must preserve point-in-time correctness, reproducibility, and honest out-of-sample evaluation.

## Non-negotiable research rules

1. **No look-ahead bias.** A feature or event may be used for a decision only if it was actually available at or before that decision timestamp. Never use future prices, future macro vintages, future event publication/availability, or later revisions.
2. **Preserve point-in-time semantics.** Keep `event_time`, `published_at`, and `available_at` distinct. Do not replace an unavailable exact timestamp with a later or future timestamp. If a source only provides a proxy, preserve and document that proxy.
3. **Chronological validation only.** Do not introduce random train/test splits for temporal market prediction. Use walk-forward evaluation with a purge gap.
4. **Purge is mandatory.** The purge gap must satisfy the prediction horizon requirements and must be applied consistently to every comparable experiment.
5. **A/B/C are one experiment family.**
   - A = technical features.
   - B = technical + macro.
   - C = technical + macro + geopolitical/events.
   - A, B and C MUST use the same observations, same target, same lockbox, same purge gap, same folds, same transaction-cost assumptions and same slippage assumptions.
   - Do not allow individual variants to construct their own folds or silently alter protocol parameters.
6. **Single fold generator.** Shared A/B/C fold construction must go through the canonical common-fold mechanism. If a change introduces another fold generator, stop and refactor it into the common protocol instead.
7. **Final OOS lockbox is sacred.** Never tune hyperparameters, feature definitions, thresholds, event filters, model selection, or protocol choices against final lockbox outcomes. The final lockbox is for evaluation, not development.
8. **No winner selection on final OOS.** Stress/sensitivity analyses may describe robustness, but must not be used to select the final model using the untouched lockbox.
9. **Financial realism.** Backtests must account for transaction costs, slippage, turnover, drawdown, volatility and risk-adjusted performance. Signals must respect the exact execution timing contract.
10. **Benchmarks are mandatory where applicable.** Compare against buy-and-hold and deterministic/random baselines on the exact same OOS observation index.
11. **Reproducibility.** Preserve dataset hashes, source manifests, protocol versions, code versions and deterministic seeds. Do not introduce nondeterministic behavior without an explicit reason and test.
12. **Data provenance.** New historical sources must record source identity, coverage, retrieval/provenance metadata and a reproducible representation/hash where practical. Do not silently mix revised and unrevised data.
13. **Historical calendars matter.** Respect exchange sessions, weekends, holidays, early closes and timezone conversions. Never assume every UTC date is a tradable market session.
14. **Macro data must be vintage-aware.** Preserve observation date separately from vintage availability. A revised macro value must not leak into an earlier decision.
15. **Events must be admitted per decision.** Never validate an entire event universe only against the latest decision timestamp. Admission must be evaluated for each decision timestamp.
16. **D is C + AI.** D must use the frozen C lockbox and identical observations/folds/protocol. AI is an incremental overlay, not a replacement for the quantitative predictor. D may only measure incremental value attributable to AI.
17. **Do not weaken tests to make code pass.** Fix implementation defects. Tests may be corrected only when their expected behavior contradicts the documented scientific contract.
18. **Prefer explicit failures.** Invalid timestamps, NaN/inf values, duplicate identifiers, mismatched indexes, protocol mismatches and missing provenance should fail loudly rather than being silently coerced.

## Coding rules

- Keep changes small, deterministic and reviewable.
- Reuse existing validators and protocol objects instead of duplicating logic.
- Before adding a new abstraction, search the repository for an existing equivalent.
- Avoid duplicate sources of truth for A/B/C protocol, folds, lockbox configuration and data timing.
- Add or update tests for every research-critical behavior.
- Preserve public APIs unless there is a strong reason to change them; if changing one, update callers and tests together.
- Use timezone-aware timestamps for market/event decision times.
- Do not silently sort, forward-fill, interpolate, or impute historical information when doing so could change point-in-time semantics. Make such transformations explicit and tested.
- Keep raw-source semantics intact; normalized datasets must retain enough metadata to audit how values were admitted.

## Review checklist for Copilot-generated changes

Before proposing a change, verify:

- Could any value originate after the decision timestamp?
- Could the change alter the final OOS lockbox or its definition?
- Could A/B/C receive different folds, rows, target definitions, or protocol parameters?
- Does the change preserve purge-gap enforcement?
- Does it preserve deterministic hashes/manifests?
- Does it preserve transaction-cost/slippage accounting?
- Does it require new tests for leakage, chronology, provenance, or reproducibility?
- If it touches D/AI, does it prove D remains C + AI on the same frozen lockbox?

If any answer is uncertain, do not silently implement the change. Flag the uncertainty and propose a safe, testable approach.

## Priority order

When instructions conflict, prioritize:

1. Point-in-time correctness and no leakage.
2. Untouched final OOS lockbox integrity.
3. Shared A/B/C experimental protocol.
4. Reproducibility and provenance.
5. Financial execution realism.
6. Tests and maintainability.
7. Convenience or implementation speed.

Copilot is a coding assistant. It must not invent scientific conclusions, select a final model from OOS results, or relax the research protocol merely to make an implementation easier.
