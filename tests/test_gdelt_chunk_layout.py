from scripts.ingest_gdelt_chunk import GDELT_EXPORT_COLUMNS_61


def test_gdelt_export_layout_keeps_date_added_aligned() -> None:
    assert len(GDELT_EXPORT_COLUMNS_61) == 61
    assert GDELT_EXPORT_COLUMNS_61[39] == "actor1_geo_adm2"
    assert GDELT_EXPORT_COLUMNS_61[47] == "actor2_geo_adm2"
    assert GDELT_EXPORT_COLUMNS_61[55] == "action_geo_adm2"
    assert GDELT_EXPORT_COLUMNS_61[59] == "date_added"
    assert GDELT_EXPORT_COLUMNS_61[60] == "source_url"
