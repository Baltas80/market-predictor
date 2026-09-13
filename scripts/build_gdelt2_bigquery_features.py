"""Build PIT-safe daily GDELT 2.0 features from the public BigQuery table.

The query is partition-pruned and aggregates the 15-minute GDELT stream before
Python applies the exact NYSE session-close cutoff and seven-day decay. This
avoids materializing millions of raw event rows while retaining all qualifying
GDELT events in the historical feature calculation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from google.cloud import bigquery

TABLE = "gdelt-bq.gdeltv2.events_partitioned"
MAX_BYTES_BILLED = 100_000_000_000  # 100 GB hard ceiling for the full historical query.
HALF_LIFE_DAYS = 7.0
UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")
CATEGORIES = (
    "war_conflict", "political_crisis", "corruption", "corporate_scandal",
    "political_scandal", "financial_fraud", "sanctions", "regulation",
    "terrorism", "social_unrest", "economic_crisis", "natural_disaster",
    "public_health",
)


def _category_case() -> str:
    return """CASE
        WHEN CAST(EventCode AS STRING) = '163' OR CAST(EventCode AS STRING) = '0163' THEN 'sanctions'
        WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') = '14' THEN 'social_unrest'
        WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') IN ('18','19','20') THEN
          CASE WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') = '20'
                    AND REGEXP_CONTAINS(LOWER(CONCAT(COALESCE(Actor1Name,''),' ',COALESCE(Actor2Name,''))), r'terror|militant|extremist')
               THEN 'terrorism' ELSE 'war_conflict' END
        WHEN LPAD(CAST(EventRootCode AS STRING), 2, '0') IN ('16','17','09','10','11','12','13','15') THEN 'political_crisis'
        ELSE 'political_crisis'
      END"""


def _query(start: str, end: str) -> str:
    category = _category_case()
    return f"""
      SELECT
        TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(DATEADDED), 900) * 900 + 899) AS available_bucket_end,
        {category} AS category,
        SUM(LEAST(1.0, 0.7 * LEAST(ABS(COALESCE(GoldsteinScale, 0.0)), 10.0) / 10.0
          + 0.3 * SAFE_DIVIDE(GREATEST(COALESCE(NumSources, 0), 0), GREATEST(COALESCE(NumSources, 0), 0) + 5.0))) AS severity_sum,
        COUNT(*) AS event_count
      FROM `{TABLE}`
      WHERE _PARTITIONDATE BETWEEN @start_date AND @end_date
        AND DATE(DATEADDED) BETWEEN @start_date AND @end_date
        AND DATEADDED IS NOT NULL
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

    rows = []
    decay_k = np.log(2.0) / HALF_LIFE_DAYS
    for close in closes:
        eligible = frame.loc[frame["available_bucket_end"] <= close]
        row: dict[str, float | str] = {"decision_time": close.isoformat()}
        if eligible.empty:
            for category in CATEGORIES:
                row[f"event_{category}"] = 0.0
            row.update(event_total_pressure=0.0, event_count=0.0, event_surprise=0.0)
            rows.append(row)
            continue
        elapsed_days = (close - eligible["available_bucket_end"]).dt.total_seconds().to_numpy() / 86400.0
        weights = np.exp(-decay_k * elapsed_days)
        pressure = eligible["severity_sum"].to_numpy() * weights
        counts = eligible["event_count"].to_numpy() * weights
        for category in CATEGORIES:
            mask = eligible["category"].to_numpy() == category
            row[f"event_{category}"] = float(pressure[mask].sum())
        row["event_total_pressure"] = float(pressure.sum())
        row["event_count"] = float(counts.sum())
        row["event_surprise"] = 0.0
        rows.append(row)
    result = pd.DataFrame(rows)
    result["decision_time"] = pd.to_datetime(result["decision_time"], utc=True)
    result = result.set_index("decision_time").sort_index()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--start", default="2015-02-19")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", default="data/historical/normalized/events_gdelt2_features.csv")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    query = _query(args.start, args.end)
    params = [
        bigquery.ScalarQueryParameter("start_date", "DATE", args.start),
        bigquery.ScalarQueryParameter("end_date", "DATE", args.end),
    ]
    config = bigquery.QueryJobConfig(
        query_parameters=params,
        maximum_bytes_billed=MAX_BYTES_BILLED,
    )
    job = client.query(query, job_config=config)
    rows = list(job.result())
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
