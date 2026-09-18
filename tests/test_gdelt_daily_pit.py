from datetime import date

import pandas as pd

from market_predictor.gdelt1 import GDELT_SOURCE_ID, gdelt1_daily_availability


def test_gdelt1_source_uses_canonical_conservative_pit_boundary():
    available = gdelt1_daily_availability(date(2015, 2, 19))
    assert available == pd.Timestamp("2015-02-20T11:00:00Z")
    assert GDELT_SOURCE_ID == "GDELT_1_Event_Database"


def test_gdelt1_pit_boundary_is_dst_aware():
    available = gdelt1_daily_availability(date(2015, 7, 3))
    assert available == pd.Timestamp("2015-07-04T10:00:00Z")
