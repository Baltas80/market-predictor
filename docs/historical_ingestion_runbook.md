# Historical ingestion runbook

## Scope

The declared research window is 2000-01-03 through 2025-12-31. The ingestion executor is **dry-run by default** and does not treat downloaded data as validated research input.

## Required sequence

1. Build the immutable coverage plan.
2. Retrieve raw source artifacts into staging only.
3. Record source URI, retrieval version, coverage, row count and SHA-256.
4. Preserve source-specific timing fields.
5. Normalize into the versioned research schema.
6. Validate market sessions and UTC/session-close semantics.
7. Validate FRED vintage ordering and retain date-level availability semantics.
8. Deduplicate events by identifier while retaining the earliest defensible availability record.
9. Reconstruct the information set separately for every decision timestamp.
10. Assemble the common A/B/C observation index and hash it.
11. Pass the Historical Gate.
12. Freeze the A/B/C protocol and lockbox.
13. Generate A/B/C predictions through `ABCExecutionPlan`.
14. Run the financial matrix and temporal stability analysis.
15. Only after C is frozen, evaluate D as C + AI.

## Market timing

Daily market bars are represented at the modeled cash-session close. The session audit rejects weekends, full-day holidays and timestamps after the applicable regular/early close.

## FRED timing

FRED `observation_date` and vintage dates are distinct. The repository does not invent an intraday release timestamp. The current aligner uses a conservative date-level session lag; this policy must be recorded in the dataset manifest.

## GDELT 1.0 timing

Production historical event ingestion uses the GDELT 1.0 daily stream under `https://data.gdeltproject.org/events/{date}.export.CSV.zip`. The archive's `DATEADDED` field may be an 8-digit archive date and is **not** treated as an availability timestamp. `SQLDATE` is the event date and is never used as availability.

For PIT gating, when the exact publication instant cannot be demonstrated, the pipeline uses the conservative next-day 06:00 US Eastern publication boundary for the archive day. This is converted to UTC with the `America/New_York` timezone so DST is handled explicitly. `published_at` remains unknown unless independently supplied. The normalized source identifier is `GDELT_1_Event_Database` and must not be conflated with the separate GDELT 2.0 stream.

## Lockbox rule

A successful download, checksum, or schema validation does **not** by itself authorize lockbox admission. The Historical Gate must pass first, and final OOS parameters cannot be tuned from lockbox outcomes.

## Executor

Run the planner without downloading:

`python scripts/ingest_historical.py`

The explicit `--execute` switch is intentionally not wired to retrieval yet. Source adapters must be reviewed and connected to staged artifact writing before real downloads can become admissible research data.
