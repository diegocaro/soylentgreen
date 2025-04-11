from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from aqara_video.core.clip import Clip
from aqara_video.core.provider import CameraProvider
from aqara_video.core.timeline import InvalidCameraDirError, Timeline


@pytest.fixture
def mock_provider() -> Mock:
    """Fixture to create a mock camera provider."""
    mock_provider = Mock(spec=CameraProvider)
    # Default behavior for validate_directory
    mock_provider.validate_directory.return_value = True
    # Default clips to return
    clips = [
        Mock(spec=Clip, timestamp=datetime(2025, 2, 7, 9, 12, 2)),
        Mock(spec=Clip, timestamp=datetime(2025, 2, 7, 10, 15, 30)),
        Mock(spec=Clip, timestamp=datetime(2025, 2, 8, 8, 5, 45)),
    ]
    mock_provider.load_clips.return_value = clips
    return mock_provider


@pytest.fixture
def timeline_data() -> Dict[str, Any]:
    """Fixture providing common test data for timeline."""
    return {
        "clips_path": Path("/test/camera/lumi1.54ef44457bc9"),
    }


def test_timeline_initialization(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test that Timeline is initialized correctly with valid inputs."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Check instance attributes
    assert timeline.clips_path == timeline_data["clips_path"]
    assert timeline.provider == mock_provider
    assert timeline.camera_id == timeline_data["clips_path"].name

    # Validate directory should be called during initialization
    mock_provider.validate_directory.assert_called_once_with(
        timeline_data["clips_path"]
    )

    # Clips should not be loaded yet (lazy loading)
    assert timeline._clips is None
    mock_provider.load_clips.assert_not_called()


def test_timeline_invalid_directory(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test that Timeline raises an exception with invalid directory."""
    # Configure the mock to indicate an invalid directory
    mock_provider.validate_directory.return_value = False

    # Should raise InvalidCameraDirError
    with pytest.raises(InvalidCameraDirError) as excinfo:
        Timeline(
            clips_path=timeline_data["clips_path"],
            provider=mock_provider,
        )

    # Validate the exception contains the path
    assert timeline_data["clips_path"] == excinfo.value.path


def test_timeline_lazy_loading(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test that clips are loaded lazily only when accessed."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Before accessing clips
    assert timeline._clips is None
    mock_provider.load_clips.assert_not_called()

    # Access clips property
    clips = timeline.clips

    # After accessing clips
    assert timeline._clips is not None
    mock_provider.load_clips.assert_called_once_with(timeline_data["clips_path"])

    # Access clips property again
    clips_again = timeline.clips

    # Should not call load_clips again
    mock_provider.load_clips.assert_called_once()

    # Should be the same clips
    assert clips is clips_again


def test_timeline_get_available_dates(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test that available dates are correctly extracted from clips."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Expected dates based on the mock clips
    expected_dates = [date(2025, 2, 7), date(2025, 2, 8)]

    # Get available dates
    available_dates = timeline.get_available_dates()

    # Check results
    assert available_dates == expected_dates


def test_timeline_search_clips_no_filter(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test searching clips without date filters."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Search without filters should return all clips
    clips = timeline.search_clips()

    # Should return all clips
    assert len(clips) == 3
    assert clips == timeline.clips


def test_timeline_search_clips_with_from_date(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test searching clips with from_date filter."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Search with from_date
    date_from = datetime(2025, 2, 7, 10, 0, 0)
    clips = timeline.search_clips(date_from=date_from)

    # Should return only clips after date_from
    assert len(clips) == 2
    assert all(clip.timestamp >= date_from for clip in clips)


def test_timeline_search_clips_with_to_date(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test searching clips with to_date filter."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Search with to_date
    date_to = datetime(2025, 2, 7, 10, 0, 0)
    clips = timeline.search_clips(date_to=date_to)

    # Should return only clips before date_to
    assert len(clips) == 1
    assert all(clip.timestamp <= date_to for clip in clips)


def test_timeline_search_clips_with_date_range(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test searching clips with both from_date and to_date filters."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Search with date range
    date_from = datetime(2025, 2, 7, 9, 30, 0)
    date_to = datetime(2025, 2, 7, 23, 59, 59)
    clips = timeline.search_clips(date_from=date_from, date_to=date_to)

    # Should return only clips within the date range
    assert len(clips) == 1
    assert all(date_from <= clip.timestamp <= date_to for clip in clips)


def test_timeline_str_representation(
    mock_provider: Mock, timeline_data: Dict[str, Any]
) -> None:
    """Test the string representation of Timeline."""
    # Create custom mock clips with specific string representations
    clips = [
        Mock(spec=Clip),
        Mock(spec=Clip),
        Mock(spec=Clip),
    ]
    # Configure the string representations
    clips[0].__str__ = Mock(return_value="Clip 1")
    clips[1].__str__ = Mock(return_value="Clip 2")
    clips[2].__str__ = Mock(return_value="Clip 3")

    # Update the provider to return our custom clips
    mock_provider.load_clips.return_value = clips

    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Expected string representation
    expected_str = "Clip 1\nClip 2\nClip 3"

    # Check string representation
    assert str(timeline) == expected_str


def test_timeline_length(mock_provider: Mock, timeline_data: Dict[str, Any]) -> None:
    """Test the length calculation of Timeline."""
    timeline = Timeline(
        clips_path=timeline_data["clips_path"],
        provider=mock_provider,
    )

    # Expected length
    expected_length = 3

    # Check length
    assert len(timeline) == expected_length
