# Multi-agent development workflow

## Purpose

This document defines how multiple Codex agents/runners can develop Market Predictor concurrently without compromising the research protocol.

## Topology

```text
                         ORCHESTRATOR
                              |
        +----------+----------+----------+----------+----------+
        |          |          |          |          |          |
     STATS     ROBUSTNESS  BENCHMARKS  REGIMES  CALIBRATION    AI
        |          |          |          |          |          |
        +----------+----------+----------+----------+----------+
                              |
                         QA / REVIEW
                              |
                         INTEGRATION
                              |
                         FINAL LOCKBOX
```

The orchestrator assigns independent tasks, keeps work isolated and collects PRs. It must not ask agents to optimize against the final lockbox.

## Worktree/branch policy

Each agent gets its own branch/worktree. Recommended names:

- `agent/statistics-*`
- `agent/robustness-*`
- `agent/benchmarks-*`
- `agent/regimes-*`
- `agent/calibration-*`
- `agent/ai-*`
- `agent/financial-*`
- `agent/qa-*`

Agents should touch only the files required for their task. Cross-cutting changes should be handed to the owning workstream or coordinated explicitly.

## Task queue

Existing GitHub research issues provide the primary queue:

- #4 — reality benchmark A/B/C/D
- #5 — overfitting and sensitivity stress suite
- #6 — feature importance and incremental value attribution
- #7 — paper-trading signal and risk monitoring
- #8 — crisis/regime evaluation

Additional parallel tracks:

- statistical robustness and confidence intervals;
- probability calibration;
- advanced financial accounting and execution assumptions;
- AI event extraction and D = C + AI.

## Dependency rules

### Can start immediately

Statistics, calibration, robustness, benchmarks, regimes, financial accounting and QA can develop against stable interfaces and synthetic fixtures without waiting for the full real-data run.

### Depends on finalized C interfaces

The AI track should consume the C event schema and preserve point-in-time availability. It should not redesign the quantitative evaluation protocol.

### Depends on frozen evaluation protocol

Final lockbox execution must happen only after the experiment definition, features, model/configuration, threshold and financial assumptions are frozen.

## Integration gate

A PR is eligible for integration only if:

1. focused tests pass;
2. full tests pass;
3. no leakage safeguard is weakened;
4. the change respects point-in-time availability;
5. documentation is updated when behavior/protocol changes;
6. the branch is based on the current integration target or has been explicitly reconciled;
7. no lockbox result was used for tuning.

## Lockbox gate

Before a final lockbox run, the orchestrator/reviewer records a frozen experiment manifest containing:

- dataset/version/hash;
- lockbox dates;
- purge horizon;
- feature set;
- model/configuration;
- threshold/decision rule;
- transaction costs/slippage;
- benchmark definitions;
- code version;
- random seeds;
- AI model and prompt/schema versions where applicable.

After execution, the lockbox is considered consumed. Any tuning triggered by its results must return to pre-lockbox validation data and produce a new experiment version rather than silently changing the final protocol.

## Autonomous runner loop

A capable Codex runner should follow this loop:

1. Pull current integration target.
2. Read `AGENTS.md` and the assigned issue.
3. Inspect relevant code/tests.
4. Create an isolated branch/worktree.
5. Implement one coherent change.
6. Add/update tests.
7. Run focused tests.
8. Run the full suite.
9. Perform a temporal/leakage sanity check when relevant.
10. Commit with a descriptive message.
11. Open/update a PR with evidence and limitations.
12. Stop and hand off; do not merge unrelated work.

## Orchestrator responsibilities

The coordinator should:

- keep the task queue moving;
- avoid duplicate work;
- enforce dependency order;
- detect stale PR bases;
- require green CI;
- request reviewer/QA checks;
- maintain the frozen lockbox manifest;
- consolidate only reviewed changes.

The coordinator should never decide that a model is better merely because it has a higher lockbox score. Statistical uncertainty, robustness, benchmarks and economic realism remain separate gates.

## Recommended first parallel wave

Launch these independent tracks first:

1. **Statistics:** finalize bootstrap/period-stability tests and add appropriate null/significance tooling.
2. **Robustness:** complete threshold/cost/slippage and model-sensitivity stress tests.
3. **Benchmarks:** harden buy-and-hold/random baselines and define the A/B/C/D comparison contract.
4. **Calibration:** test Brier/ECE/binning edge cases and define reliability reporting.
5. **Regimes:** finalize causal regime labels and crisis/non-crisis evaluation.
6. **Financial:** finish explicit total-cost/equity accounting and tests.
7. **AI:** harden the structured AI contract and availability audit before any D experiment.
8. **QA:** continuously review the above for leakage, stale branches and regression risk.

Only after these tracks converge should the final A/B/C lockbox be consumed, followed by the incremental D experiment.
