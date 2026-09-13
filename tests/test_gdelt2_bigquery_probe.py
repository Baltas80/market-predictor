from pathlib import Path


def test_gdelt2_probe_is_partition_pruned_and_fail_closed():
    text = Path("scripts/probe_gdelt2_bigquery.py").read_text(encoding="utf-8")
    assert "events_partitioned" in text
    assert "_PARTITIONDATE = @partition_date" in text
    assert "maximum_bytes_billed" in text
    assert "GLOBALEVENTID" in text
    assert "DATEADDED" in text
