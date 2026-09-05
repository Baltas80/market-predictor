# Historical staging protocol

The 2000-2025 staging executor is `scripts/ingest_historical.py` and is dry-run by default.

## Source adapters

- **Market:** Stooq SPX daily bars, represented at the modeled cash-session close in UTC.
- **Macro:** FRED `DFF`, `FEDFUNDS`, `DGS10`, `CPIAUCSL`, `UNRATE`, `VIXCLS`, retaining realtime vintages.
- **Events:** GDELT daily event exports from 2015-01-01, retaining `DATEADDED` as an availability proxy.
- **SEC:** Litigation Releases RSS snapshot. This is not a verified 2000-2025 archive and is therefore recorded as a source limitation.

## Staging phases

1. `download` — invoke the adapters.
2. `raw` — persist deterministic source snapshots.
3. `hash` — create source manifests with deterministic hashes and coverage.
4. `normalize` — persist the definitive market/macro/event schemas.
5. `pit` — preserve and validate information-availability rules, including FRED vintages.
6. `historical_gate` — fail closed unless the complete staged inputs pass schema, session, PIT and manifest validation.
7. `dataset` — only after the gate passes is the staged dataset marked admissible.

No download result is automatically a lockbox result.

## Reproducibility

Each staging output records `STAGING_VERSION`, source coverage, row counts, SHA-256 manifests and the exact source availability policy. The executor writes `ingestion_plan.json`, `source_manifest.json` and `staging_result.json`.

`FRED_API_KEY` must be supplied through the environment for real execution. The default command performs no network retrieval.

For a full historical run, use `--execute`. GDELT retrieval is intentionally explicit because it requires one daily export per calendar day over its available historical range. `--skip-gdelt` and `--skip-sec` are available for staged source-by-source validation and do not imply complete event coverage.
