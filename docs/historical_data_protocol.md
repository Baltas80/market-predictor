# Historical data contract

## Time semantics

Every exogenous event uses three distinct timestamps:

- `event_time`: when the underlying event occurred.
- `published_at`: when the source published the information.
- `available_at`: when the information is admitted to the model information set.

Only `available_at` gates event features. `event_time` must not be substituted for publication time, and `published_at` must not be substituted for availability when a source provides a stricter availability rule.

Macro observations use `observation_date` plus FRED `vintage_start`/`vintage_end`. A model decision may use only the vintage that was available by that decision cutoff.

Market observations use one unique, chronological timezone-aware decision index and validated OHLCV values.

## Provenance and duplicates

Each source is identified by `source_id`. File provenance records include source, retrieval method/time, byte size and SHA-256. Event identities are deduplicated by `event_id`; macro vintages are unique by `(series_id, observation_date, vintage_start)`; market timestamps are unique by decision timestamp.

A missing historical source date is an ingestion failure, not an implicit empty observation.

## Common observations

A/B/C are assembled from the same validated market observation universe. Feature groups may differ, but the observation index, target horizon, folds, purge gap, lockbox identity and execution assumptions may not.
