"""Download and parse GDELT 2.0 daily event exports."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from .gdelt import normalize_gdelt_events

GDELT_EVENTS_URL = "https://data.gdeltproject.org/events/{date:%Y%m%d}.export.CSV.zip"
GDELT_EVENT_COLUMNS = [
    "GlobalEventID", "SQLDATE", "MonthYear", "Year", "FractionDate", "Actor1Code", "Actor1Name",
    "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code",
    "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type",
    "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName",
    "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_daily_event_file(event_date: date, destination: str | Path, *, timeout: int = 60) -> dict[str, object]:
    """Download one immutable raw daily GDELT export and return provenance."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = GDELT_EVENTS_URL.format(date=event_date)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return {"url": url, "path": str(destination), "sha256": sha256_file(destination), "bytes": destination.stat().st_size}


def read_daily_event_file(path: str | Path) -> pd.DataFrame:
    """Read the selected fields from a GDELT daily ZIP export."""
    path = Path(path)
    with ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"Expected one CSV member, found {len(members)}")
        with archive.open(members[0]) as handle:
            raw = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=GDELT_EVENT_COLUMNS,
                usecols=["GlobalEventID", "SQLDATE", "DATEADDED", "EventCode", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "AvgTone", "SOURCEURL"],
                dtype={"GlobalEventID": "string", "SQLDATE": "string", "DATEADDED": "string", "EventCode": "string", "SOURCEURL": "string"},
            )
    return normalize_gdelt_events(raw)
