from scripts.ingest_gdelt_chunk import GDELT_EXPORT_COLUMNS


def test_gdelt_export_layout_keeps_date_added_aligned() -> None:
    assert len(GDELT_EXPORT_COLUMNS) == 61
    assert GDELT_EXPORT_COLUMNS[39] == "actor1_geo_adm2"
    assert GDELT_EXPORT_COLUMNS[47] == "actor2_geo_adm2"
    assert GDELT_EXPORT_COLUMNS[55] == "action_geo_adm2"
    assert GDELT_EXPORT_COLUMNS[59] == "date_added"
    assert GDELT_EXPORT_COLUMNS[60] == "source_url"
