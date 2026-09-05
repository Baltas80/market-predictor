# Deep temporal audit protocol

## Market sessions

Daily market observations are mapped to the real US cash-market session calendar. UTC timestamps are converted to `America/New_York` before deciding whether a record belongs to a session, occurs before the regular close, occurs after the close, or falls on a weekend/holiday. Early-close sessions use their actual scheduled close.

## Macro

A macro observation has two distinct clocks: `observation_date` and vintage availability. The model may use a value only when the vintage was available by the decision cutoff. Date-only release metadata is treated conservatively; it is never upgraded to an invented intraday timestamp.

## GDELT

GDELT `DATEADDED` is an availability/database-ingestion proxy, not an article publication timestamp. The normalized event keeps:

- `event_time`: GDELT event date;
- `published_at`: unknown unless a source provides it;
- `available_at`: `DATEADDED` proxy;
- `availability_proxy`: `DATEADDED`.

This prevents `DATEADDED` from being falsely represented as publication time.

## After-close rule

Information arriving after the market close cannot affect that day's close-based decision. It becomes eligible only for the next admissible decision according to the execution contract.

## Required audit outputs

The ingestion audit must report coverage, missing sessions, duplicate IDs, invalid timestamps, future information, weekend/holiday records, after-close publications, source failures and provenance hashes. A failed required-source audit blocks downstream lockbox evaluation.
