"""FRED/ALFRED point-in-time economic data client."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import requests

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


@dataclass(frozen=True)
class FredClient:
    """Small FRED client with explicit real-time/vintage controls.

    The API key is read from ``FRED_API_KEY`` unless supplied explicitly.
    ``realtime_end`` is intentionally explicit so backtests can request the
    information that was available at a historical point in time.
    """

    api_key: str | None = None
    timeout_seconds: int = 30

    def _key(self) -> str:
        key = self.api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED_API_KEY is required to access FRED/ALFRED")
        return key

    def observations(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        vintage_dates: str | None = None,
        units: str = "lin",
    ) -> pd.DataFrame:
        """Return observations plus FRED real-time metadata."""
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._key(),
            "file_type": "json",
            "sort_order": "asc",
            "units": units,
        }
        optional = {
            "observation_start": observation_start,
            "observation_end": observation_end,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "vintage_dates": vintage_dates,
        }
        params.update({k: v for k, v in optional.items() if v is not None})

        response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("observations", [])
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])

        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        for column in ("realtime_start", "realtime_end"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def point_in_time(
        self,
        series_id: str,
        prediction_times: pd.DatetimeIndex,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.DataFrame:
        """Get a conservative point-in-time panel using ALFRED real-time dates.

        One request is made per unique prediction date. This favors correctness
        and reproducibility over throughput; production jobs can cache vintages.
        """
        times = pd.DatetimeIndex(pd.to_datetime(prediction_times, utc=True)).sort_values().unique()
        if len(times) == 0:
            return pd.DataFrame(columns=["prediction_time", "date", "value"])

        frames: list[pd.DataFrame] = []
        for timestamp in times:
            vintage = timestamp.date().isoformat()
            observations = self.observations(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end,
                realtime_start=vintage,
                realtime_end=vintage,
            )
            if observations.empty:
                continue
            observations = observations.copy()
            observations["prediction_time"] = timestamp
            frames.append(observations[["prediction_time", "date", "value"]])

        if not frames:
            return pd.DataFrame(columns=["prediction_time", "date", "value"])
        return pd.concat(frames, ignore_index=True)
