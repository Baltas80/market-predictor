from __future__ import annotations

import pandas as pd

from scripts.ingest_gdelt_chunk import _parse_gdelt_date_added


def test_parse_gdelt_date_added_accepts_canonical_integer_strings() -> None:
    values = pd.Series(["20150219000000", "20150219153000"])

    result = _parse_gdelt_date_added(values)

    assert result.tolist() == [
        pd.Timestamp("2015-02-19T00:00:00Z"),
        pd.Timestamp("2015-02-19T15:30:00Z"),
    ]


def test_parse_gdelt_date_added_normalizes_numeric_like_encodings() -> None:
    values = pd.Series(["20150219000000.0", "2.0150219153e13"])

    result = _parse_gdelt_date_added(values)

    assert result.tolist() == [
        pd.Timestamp("2015-02-19T00:00:00Z"),
        pd.Timestamp("2015-02-19T15:30:00Z"),
    ]


def test_parse_gdelt_date_added_rejects_invalid_values_without_imputation() -> None:
    values = pd.Series(["not-a-date", None])

    result = _parse_gdelt_date_added(values)

    assert result.isna().all()
