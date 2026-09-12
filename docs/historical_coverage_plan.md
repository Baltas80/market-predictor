# Historical dataset coverage plan

## Effective coverage

The first reproducible research corpus is defined as **2000-01-03 through 2025-12-31**. The endpoint is a completed calendar year so the final OOS lockbox is not contaminated by an incomplete current period.

| Source | Coverage | Role | Point-in-time rule |
|---|---|---|---|
| Stooq S&P 500 daily | 2000-01-03 → 2025-12-31 | Market OHLCV | Session date; daily close interpreted through the US cash-market calendar |
| FRED FEDFUNDS | 2000-01-03 → 2025-12-31 | Effective federal funds rate | Preserve realtime/vintage fields; no unreleased revision may enter a session |
| FRED DGS10 | 2000-01-03 → 2025-12-31 | 10Y Treasury yield | Same vintage rule |
| FRED CPIAUCSL | 2000-01-03 → 2025-12-31 | Inflation | Vintage-aware; release time is not inferred from observation date |
| FRED UNRATE | 2000-01-03 → 2025-12-31 | Labor market | Vintage-aware; release time is not inferred from observation date |
| FRED VIXCLS | 2000-01-03 → 2025-12-31 | Volatility context | Daily market observation |
| GDELT 2.0 Events | 2015-02-19 → 2025-12-31 | Geopolitical/events | `event_time`, publication and `available_at` remain separate; `DATEADDED` is an explicit availability proxy |
| SEC Litigation Releases | 2000-01-01 → current feed snapshot | Regulatory/corporate events | Feed publication timestamp is the conservative daily availability marker; this is not a verified 2000-2025 archive |

Raw source files are not intended to be committed to Git. The ingestion layer must record source URL, retrieval metadata, row count, coverage, and SHA-256 in immutable manifests.

## Session audit

Market timestamps are normalized to UTC and mapped to `America/New_York`. A regular cash-market close is 16:00 ET; principal early closes are modeled at 13:00 ET. Weekends, full-day holidays and early-close sessions are explicitly represented. A timestamp after the expected close is never silently treated as information available at that session's close.

FRED date-level vintages remain conservative until an exact release timestamp is available. GDELT `DATEADDED` is treated as an availability proxy, not as proof of article publication time. Any proxy must remain visible in provenance.

## Lockbox rule

The coverage window is a data boundary, not a model-selection mechanism. Lockbox dates, purge gap, feature set, threshold and financial assumptions must be frozen before final OOS evaluation. A/B/C consume the same observations and the same fold sequence. Required GDELT source-days must be fully resolved before the final lockbox; unresolved days cause the staging gate to fail closed.
