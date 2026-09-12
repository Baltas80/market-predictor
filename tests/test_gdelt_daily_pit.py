from datetime import date

import pandas as pd

from scripts.ingest_gdelt_daily_chunk import GDELT_SOURCE_ID, _availability_for_file_date


def test_gdelt1_source_uses_conservative_next_day_boundary():
    available = _availability_for_file_date(date(2015, 2, 19))
    assert available == pd.Timestamp("2015-02-20T12:00:00Z")
    assert GDELT_SOURCE_ID == "GDELT_1_Daily_Event_Database"
