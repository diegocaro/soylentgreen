from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from aqara_video.providers.aqara import AqaraProvider


@pytest.mark.parametrize("tz_name", ["UTC", "America/Santiago"])
def test_extract_timestamp_timezone(monkeypatch, tz_name):
    monkeypatch.setenv("AQARA_TIMEZONE", tz_name)
    provider = AqaraProvider()
    path = Path("lumi1.54ef44457bc9/20250207/082900.mp4")
    ts = provider.extract_timestamp(path)
    expected_dt = datetime(2025, 2, 7, 8, 29, 0, tzinfo=ZoneInfo(tz_name))
    assert ts == expected_dt


# Optionally, add more tests for different timezones
