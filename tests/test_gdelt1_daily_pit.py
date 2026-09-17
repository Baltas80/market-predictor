from datetime import timedelta

import pandas as pd

from scripts.ingest_gdelt_chunk import GDELT_SOURCE_ID, gdelt1_daily_availability


def test_gdelt1_daily_availability_uses_next_day_publication_boundary():
    available = gdelt1_daily_availability("2025-01-01")
    assert available == pd.Timestamp("2025-01-02T11:00:00Z")


def test_gdelt1_daily_availability_handles_est_dst():
    available = gdelt1_daily_availability("2025-07-01")
    assert available == pd.Timestamp("2025-07-02T10:00:00Z")


def test_gdelt1_source_id_is_not_gdelt2():
    assert GDELT_SOURCE_ID == "GDELT_1_Event_Database"
    assert GDELT_SOURCE_ID != "GDELT_2_Event_Database"
