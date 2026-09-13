"""Build PIT-safe daily GDELT 2.0 features from public BigQuery data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

TABLE = "gdelt-bq.gdeltv2.events_partitioned"
MAX_BYTES_BILLED = 100_000_000_000
HALF_LIFE_DAYS = 7.0
CATEGORIES = (
    "war_conflict", "political_crisis", "corruption", "corporate_scandal",
    "political_scandal", "financial_fraud", "sanctions", "regulation",
    "terrorism", "social_unrest", "economic_crisis", "natural_disaster",
    "public_health",
)


def _category_case() -> str:
    return """CASE
      WHEN CAST(EventCode AS STRING) IN ('163','0163') THEN 'sanctions'
      WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') = '14' THEN 'social_unrest'
      WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') IN ('18','19','20') THEN
        CASE WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') = '20'
          AND REGEXP_CONTAINS(LOWER(CONCAT(COALESCE(Actor1Name,''),' ',COALESCE(Actor2Name,''))), r'terror|militant|extremist')
          THEN 'terrorism' ELSE 'war_conflict' END
      WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') IN ('16','17','09','10','11','12','13','15') THEN 'political_crisis'
      ELSE 'political_crisis' END"""


def _query() -> str:
    category = _category_case()
    return f"""
      WITH base AS (
        SELECT
          PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)) AS added_at,
          {category} AS category,
          LEAST(1.0, 0.7 * LEAST(ABS(COALESCE(GoldsteinScale, 0.0)), 10.0) / 10.0
            + 0.3 * SAFE_DIVIDE(GREATEST(COALESCE(NumSources, 0), 0), GREATEST(COALESCE(NumSources, 0), 0) + 5.0)) AS severity
        FROM `{TABLE}`
        WHERE _PARTITIONDATE BETWEEN @start_date AND @end_date
          AND DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING))) BETWEEN @start_date AND @end_date
          AND DATEADDED IS NOT NULL
      )
      SELECT
        TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(added_at), 900) * 900 + 899) AS available_bucket_end,
        category,
        SUM(severity) AS severity_sum,
        COUNT(*) AS event_count
      FROM base
      GROUP BY available_bucket_end, category
      ORDER BY available_bucket_end, category
    """


def _session_closes(start: str, end: str) -> pd.DatetimeIndex:
    from market_predictor.session_calendar import session_table
    table = session_table(start, end)
    return pd.DatetimeIndex(pd.to_datetime(table["close_utc"], utc=True))


def build_features(frame: pd.DataFrame, closes: pd.DatetimeIndex) -> pd.DataFrame:
    frame["available_bucket_end"] = pd.to_datetime(frame["available_bucket_end"], utc=True)
    frame["severity_sum"] = pd.to_numeric(frame["severity_sum"], errors="coerce").fillna(0.0)
    frame["event_count"] = pd.to_numeric(frame["event_count"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("available_bucket_end")
    timeline = pd.date_range(frame["available_bucket_end"].min(), frame["available_bucket_end"].max(), freq="15min", tz="UTC")
    wide_pressure = frame.pivot_table(index="available_bucket_end", columns="category", values="severity_sum", aggfunc="sum", fill_value=0.0).reindex(timeline, fill_value=0.0)
    wide_count = frame.pivot_table(index="available_bucket_end", columns="category", values="event_count", aggfunc="sum", fill_value=0.0).reindex(timeline, fill_value=0.0)
    pressure_out = pd.DataFrame(0.0, index=timeline, columns=CATEGORIES)
    count_out = pd.DataFrame(0.0, index=timeline, columns=CATEGORIES)
    decay = 2.0 ** (-(15.0 / 1440.0) / HALF_LIFE_DAYS)
    for category in CATEGORIES:
        source_pressure = wide_pressure.get(category, pd.Series(0.0, index=timeline)).to_numpy(dtype=float)
        source_count = wide_count.get(category, pd.Series(0.0, index=timeline)).to_numpy(dtype=float)
        p = np.empty(len(timeline), dtype=float)
        c = np.empty(len(timeline), dtype=float)
        prev_p = prev_c = 0.0
        for i in range(len(timeline)):
            prev_p = source_pressure[i] + prev_p * decay
            prev_c = source_count[i] + prev_c * decay
            p[i], c[i] = prev_p, prev_c
        pressure_out[category] = p
        count_out[category] = c
    selected = pressure_out.reindex(closes, method="ffill").fillna(0.0)
    selected_count = count_out.reindex(closes, method="ffill").fillna(0.0)
    selected.columns = [f"event_{c}" for c in selected.columns]
    selected["event_total_pressure"] = selected.sum(axis=1)
    selected["event_count"] = selected_count.sum(axis=1)
    selected["event_surprise"] = 0.0
    selected.index.name = "decision_time"
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--start", default="2015-02-19")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", default="data/gdelt2/events_gdelt2_features.csv")
    args = parser.parse_args()
    client = bigquery.Client(project=args.project)
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", args.start),
            bigquery.ScalarQueryParameter("end_date", "DATE", args.end),
        ],
        maximum_bytes_billed=MAX_BYTES_BILLED,
    )
    rows = list(client.query(_query(), job_config=config).result())
    raw = pd.DataFrame([dict(row) for row in rows])
    if raw.empty:
        raise RuntimeError("GDELT2 BigQuery returned no event buckets")
    closes = _session_closes(args.start, args.end)
    features = build_features(raw, closes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, date_format="%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "source": TABLE,
        "start": args.start,
        "end": args.end,
        "partition_filter": "_PARTITIONDATE BETWEEN @start_date AND @end_date",
        "availability_field": "DATEADDED",
        "availability_bucket": "15-minute bucket end (+899s)",
        "decision_cutoff": "NYSE regular session close UTC",
        "half_life_days": HALF_LIFE_DAYS,
        "rows_from_bigquery": len(raw),
        "sessions": len(features),
        "max_bytes_billed": MAX_BYTES_BILLED,
    }
    (output.parent / "gdelt2_feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
