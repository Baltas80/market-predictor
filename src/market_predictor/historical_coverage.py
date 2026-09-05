"""Declarative coverage plan for the real historical research dataset.

The plan is intentionally separate from downloaded data. It defines the
periods that must be present before an A/B/C lockbox run is admissible and
records which sources are required or optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SourceCoverage:
    source_id: str
    dataset: str
    start: date
    end: date
    required: bool
    point_in_time: bool
    notes: str = ""


# Core window: long enough to contain multiple crises while ending on a
# complete calendar year. The final lockbox itself is selected inside this
# immutable window; it is not tuned from these dates.
DATASET_START = date(2000, 1, 3)
DATASET_END = date(2025, 12, 31)

MARKET_SOURCE = SourceCoverage(
    "stooq_spx_daily",
    "S&P 500 daily OHLCV",
    DATASET_START,
    DATASET_END,
    True,
    False,
    "Daily market bars; executable-trading interpretation remains separate from the cash index.",
)

FRED_SOURCES = tuple(
    SourceCoverage("fred_vintages", series, DATASET_START, DATASET_END, True, True, "FRED realtime/vintage fields required; exact release time is not assumed.")
    for series in (
        "DFF", "FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS"
    )
)

# GDELT is deliberately split from the market/macro core. Its raw daily
# export is large, so ingestion is event-filtered after preserving source
# hashes and DATEADDED availability metadata.
GDELT_SOURCE = SourceCoverage(
    "gdelt_events",
    "GDELT 2.0 Events",
    date(2015, 1, 1),
    DATASET_END,
    True,
    True,
    "DATEADDED is retained as an availability proxy; event_time, publication and availability are never conflated.",
)

SEC_SOURCE = SourceCoverage(
    "sec_litigation_rss",
    "SEC Litigation Releases RSS",
    date(2000, 1, 1),
    DATASET_END,
    False,
    True,
    "Publication timestamp is the conservative daily availability timestamp supplied by the feed.",
)

SOURCE_COVERAGE = (MARKET_SOURCE, *FRED_SOURCES, GDELT_SOURCE, SEC_SOURCE)


def coverage_plan() -> tuple[SourceCoverage, ...]:
    """Return the immutable source/period plan in deterministic order."""
    return SOURCE_COVERAGE


def required_sources() -> tuple[SourceCoverage, ...]:
    return tuple(source for source in SOURCE_COVERAGE if source.required)


def coverage_dict() -> list[dict[str, object]]:
    """Serialize coverage for manifests/reports without timestamps or hashes."""
    return [
        {
            "source_id": source.source_id,
            "dataset": source.dataset,
            "start": source.start.isoformat(),
            "end": source.end.isoformat(),
            "required": source.required,
            "point_in_time": source.point_in_time,
            "notes": source.notes,
        }
        for source in coverage_plan()
    ]
