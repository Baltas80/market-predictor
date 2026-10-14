# Historical staging protocol

The 2000-2025 staging executor is `scripts/ingest_historical.py` and is dry-run by default.

## Source adapters

- **Market:** Stooq SPX daily bars, represented at the modeled cash-session close in UTC.
- **Macro:** FRED `FEDFUNDS`, `DGS10`, `CPIAUCSL`, `UNRATE`, `VIXCLS`, retaining realtime vintages. `DFF` is intentionally excluded from the strict PIT staging set.
- **Events:** GDELT 2.0 daily event exports from 2015-02-19, retaining `DATEADDED` as an availability proxy; publication time remains unknown unless a source supplies it.
- **SEC:** Litigation Releases RSS snapshot. This is not a verified 2000-2025 archive and is therefore recorded as a source limitation.

## Staging phases

1. `download` — invoke the adapters.
2. `raw` — persist deterministic source snapshots.
3. `hash` — create source manifests with deterministic hashes and coverage.
4. `normalize` — persist the definitive market/macro/event schemas.
5. `pit` — preserve and validate information-availability rules, including FRED vintages and GDELT `DATEADDED`.
6. `historical_gate` — fail closed unless the complete staged inputs pass schema, session, PIT and manifest validation.
7. `dataset` — only after the gate passes is the staged dataset marked admissible.

No download result is automatically a lockbox result.

## Reproducibility

Each staging output records `STAGING_VERSION`, source coverage, row counts, SHA-256 manifests and the exact source availability policy. The executor is intended to write `ingestion_plan.json`, `source_manifest.json` and `staging_result.json` when the corresponding stages are implemented and executed.

`FRED_API_KEY` must be supplied through the environment for real FRED retrieval. The default command performs no network retrieval.

## Execution status and safety

**Do not assume that `--execute` performs real downloads.** The historical ingestion runbook records that the explicit execution switch was not wired to retrieval at the time of that review. Confirm the current implementation of `scripts/ingest_historical.py` and the workflow logs before treating any staged artifact as retrieved data. This protocol document alone does not prove execution capability.

`--skip-gdelt` and `--skip-sec`, if supported by the current executor, only permit source-by-source validation and do not imply complete event coverage.

The final lockbox is fail-closed on unresolved required GDELT source-days. Missing days are never imputed or silently tolerated.
