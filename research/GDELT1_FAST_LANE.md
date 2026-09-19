# GDELT 1.0 — 9-day fast lane

## Scope

Fast-lane validation window: **2026-09-09 through 2026-09-17 inclusive** (9 calendar days).
This lane is intentionally parallel to the slower historical recovery/validation path.
It does not replace the historical path and does not alter the lockbox.

## Source verification

The official GDELT Events index currently lists all nine daily GDELT 1.0 event files and their expected MD5 hashes. The nine listed compressed sizes sum to approximately 58.3 MB.

See `research/gdelt1_9day_fast_lane_manifest.csv` for the immutable source manifest.

## Fast lane procedure

1. Resolve the exact nine official file URLs from the manifest.
2. Download each file independently; never silently skip a missing day.
3. Verify the downloaded byte stream against the manifest MD5 before extraction.
4. Extract into a date-scoped immutable directory.
5. Validate the GDELT 1.0 tab-delimited schema against the canonical codebook.
6. Count raw rows and valid rows per day.
7. Detect duplicate `GlobalEventID` values and exact duplicate event records separately.
8. Quarantine malformed rows instead of silently repairing them.
9. Record SHA/MD5, row counts, quarantine counts, duplicate counts, and checkpoint status.
10. Normalize only after source validation.
11. Feed only the canonical GDELT 1.0 event block into production C; GDELT 2.0 remains excluded.
12. Run the focused production-C tests and then the broader suite.

## Fast vs slow path

| Property | Fast lane | Slow historical lane |
|---|---|---|
| Window | 9 days | Full required historical coverage |
| Purpose | Rapid pipeline/data-path validation | Scientific historical validation |
| Source | GDELT 1.0 daily Events | GDELT 1.0 daily Events/backfile as applicable |
| Integrity | MD5 + schema + row audit | Same, plus longitudinal gap/recovery audit |
| Modeling | May unblock C/D infrastructure work | Required for final conclusions |
| Lockbox | Never touched | Final lockbox only after protocol freeze |
| Status | Intermediate research evidence | Final historical evidence |

## Point-in-time constraint

`SQLDATE`/event date is an event-time field, not proof of information availability. The production feature builder must use the project's conservative availability rule and must fail closed when availability cannot be established.

## Current execution status

**Source verification: COMPLETE for the nine requested dates.**

**Runtime download/extraction: BLOCKED in this execution environment.** The runtime network path cannot resolve `data.gdeltproject.org`, so no row count, extraction result, duplicate count, or local file hash is claimed here. This is deliberately recorded as incomplete rather than SUCCESS.

Once the files are accessible to the project runtime, the nine-day lane can proceed independently of the slow historical recovery path.
