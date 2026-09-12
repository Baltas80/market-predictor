"""Query GDELT 2.0 Events from BigQuery without downloading the raw archive.

DATEADDED is the information-availability timestamp; SQLDATE is the event
 date and is never used as availability. Only normalized rows leave BigQuery.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from market_predictor.data_sources import _gdelt_category
from market_predictor.research_schema import deduplicate_events, normalize_event_sources

SOURCE_ID = "GDELT_2_Event_Database_BigQuery"
TABLE = "`gdelt-bq.gdeltv2.events_partitioned`"

SELECT_FIELDS = """
    GLOBALEVENTID, SQLDATE, Actor1Name, Actor1CountryCode, Actor2Name,
    Actor2CountryCode, EventCode, EventRootCode, GoldsteinScale, NumMentions,
    NumSources, NumArticles, AvgTone, ActionGeo_CountryCode,
    ActionGeo_FullName, DATEADDED, SOURCEURL
"""


def _query(client, start: date, end: date) -> pd.DataFrame:
    from google.cloud import bigquery

    sql = f"""
    SELECT {SELECT_FIELDS}
    FROM {TABLE}
    WHERE _PARTITIONDATE BETWEEN @start_date AND @end_date
      AND DATEADDED IS NOT NULL
      AND SQLDATE IS NOT NULL
      AND SQLDATE <= CAST(FORMAT_DATE('%Y%m%d', @end_date) AS INT64)
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start.isoformat()),
            bigquery.ScalarQueryParameter("end_date", "DATE", end.isoformat()),
        ]
    )
    return client.query(sql, job_config=config).result().to_dataframe(create_bqstorage_client=False)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "event_id", "event_time", "published_at", "available_at", "source_id",
            "category", "severity", "country", "entity", "sector", "duration_days",
            "media_intensity", "surprise",
        ])

    frame = frame.rename(columns={
        "GLOBALEVENTID": "event_id", "SQLDATE": "sql_date",
        "Actor1Name": "actor1_name", "Actor1CountryCode": "actor1_country",
        "Actor2Name": "actor2_name", "Actor2CountryCode": "actor2_country",
        "EventCode": "event_code", "EventRootCode": "event_root_code",
        "GoldsteinScale": "goldstein_scale", "NumMentions": "num_mentions",
        "NumSources": "num_sources", "NumArticles": "num_articles",
        "AvgTone": "avg_tone", "ActionGeo_CountryCode": "action_geo_country",
        "ActionGeo_FullName": "action_geo_fullname", "DATEADDED": "date_added",
        "SOURCEURL": "source_url",
    })
    frame["event_id"] = frame["event_id"].astype("string")
    frame["event_time"] = pd.to_datetime(
        frame["sql_date"].astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d", utc=True, errors="coerce")
    frame["available_at"] = pd.to_datetime(
        frame["date_added"].astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    frame["published_at"] = pd.NaT
    for column in ("goldstein_scale", "num_mentions", "num_sources", "num_articles", "avg_tone"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    actor_text = frame[["actor1_name", "actor2_name", "action_geo_fullname"]].fillna("").agg(" ".join, axis=1)
    frame["category"] = [
        _gdelt_category(root, code, text)
        for root, code, text in zip(frame["event_root_code"], frame["event_code"], actor_text)
    ]
    scale = frame["goldstein_scale"].abs().clip(0, 10) / 10
    media = frame["num_sources"].fillna(0).clip(lower=0)
    frame["severity"] = (0.7 * scale + 0.3 * (media / (media + 5)).clip(0, 1)).clip(0, 1)
    frame["surprise"] = 0.0
    frame["country"] = frame["action_geo_country"]
    frame["entity"] = actor_text.str.slice(0, 240)
    frame["sector"] = ""
    frame["duration_days"] = 1.0
    frame["media_intensity"] = frame["num_mentions"].fillna(0).clip(lower=0).astype(float)

    normalized = normalize_event_sources(frame, source_id=SOURCE_ID)
    normalized = normalized.dropna(subset=["event_id", "event_time", "available_at", "severity"]).copy()
    return normalized[normalized["available_at"] >= normalized["event_time"]].copy()


def fetch_range(client, start: date, end: date, output: Path, missing_output: Path) -> None:
    frame = _normalize(_query(client, start, end))
    if frame.empty:
        raise RuntimeError(f"BigQuery returned no valid GDELT 2 rows for {start} -> {end}")
    frame = deduplicate_events(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    pd.DataFrame(columns=["date", "source_id", "status", "error_type", "error"]).to_csv(
        missing_output, index=False, lineterminator="\n")
    print(f"GDELT 2 BigQuery chunk saved: {start} -> {end}; rows={len(frame)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--missing-output", required=True)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()
    from google.cloud import bigquery
    client = bigquery.Client(project=args.project)
    fetch_range(client, pd.Timestamp(args.start).date(), pd.Timestamp(args.end).date(), Path(args.output), Path(args.missing_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
