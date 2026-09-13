"""Minimal, fail-closed GDELT 2.0 BigQuery connectivity probe."""
from __future__ import annotations

import argparse
from google.cloud import bigquery


TABLE = "gdelt-bq.gdeltv2.events_partitioned"
MAX_PROBE_BYTES = 100_000_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--partition-date", default="2015-02-19")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    query = f"""
        SELECT GLOBALEVENTID, DATEADDED
        FROM `{TABLE}`
        WHERE _PARTITIONDATE = @partition_date
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("partition_date", "DATE", args.partition_date)
        ],
        maximum_bytes_billed=MAX_PROBE_BYTES,
    )

    dry_run_config = bigquery.QueryJobConfig(
        query_parameters=job_config.query_parameters,
        dry_run=True,
        use_query_cache=False,
        maximum_bytes_billed=MAX_PROBE_BYTES,
    )
    dry_run = client.query(query, job_config=dry_run_config)
    print(f"dry_run_bytes={dry_run.total_bytes_processed}")

    rows = list(client.query(query, job_config=job_config).result())
    print(f"rows_returned={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
