from datetime import datetime, timezone

import pytest

from aqara_video.cli.timelapse import parse_datetime


class TestParseDateTime:
    def test_empty_string(self):
        """Test that empty strings return None."""
        assert parse_datetime("") is None
        assert parse_datetime(None) is None

    def test_date_only_with_dashes(self):
        """Test YYYY-MM-DD format."""
        dt = parse_datetime("2023-04-15")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 4
        assert dt.day == 15
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo is not None  # Should be timezone-aware

    def test_date_only_without_dashes(self):
        """Test YYYYMMDD format."""
        dt = parse_datetime("20230415")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 4
        assert dt.day == 15
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo is not None  # Should be timezone-aware

    def test_partial_datetime(self):
        """Test partial datetime strings (adding padding)."""
        # Just date and hour
        dt = parse_datetime("20230415-14")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 4
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 0
        assert dt.second == 0

        # Date, hour, and minutes
        dt = parse_datetime("20230415-1430")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 4
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 0

    def test_full_datetime(self):
        """Test full YYYYMMDD-HHMMSS format."""
        dt = parse_datetime("20230415-143022")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 4
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 22
        assert dt.tzinfo is not None  # Should be timezone-aware

    def test_whitespace_handling(self):
        """Test handling of whitespace."""
        dt = parse_datetime("  20230415-143022  ")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 4
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 22

    def test_trailing_dash_handling(self):
        """Test handling of trailing dashes."""
        dt = parse_datetime("20230415-")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 4
        assert dt.day == 15
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0

    def test_timezone_awareness(self):
        """Test that returned datetime is timezone-aware."""
        dt = parse_datetime("20230415-143022")
        assert dt.tzinfo is not None
        # The specific timezone will depend on the system's local timezone,
        # so we just verify it's timezone-aware
