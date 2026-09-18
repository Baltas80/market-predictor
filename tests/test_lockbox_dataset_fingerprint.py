
from pathlib import Path

from scripts.run_real_lockbox import _staging_fingerprint


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_dataset_fingerprint_ignores_nonproduction_sec_input(tmp_path: Path):
    _write(tmp_path / "normalized" / "market.csv", "market")
    _write(tmp_path / "raw" / "macro_fred.csv", "macro")
    _write(tmp_path / "normalized" / "events_gdelt.csv", "gdelt")
    first = _staging_fingerprint(tmp_path)

    _write(tmp_path / "normalized" / "events_sec_litigation.csv", "sec-v1")
    assert _staging_fingerprint(tmp_path) == first

    _write(tmp_path / "normalized" / "events_gdelt.csv", "gdelt-v2")
    assert _staging_fingerprint(tmp_path) != first
