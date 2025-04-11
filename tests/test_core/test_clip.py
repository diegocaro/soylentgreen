from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Protocol
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

from aqara_video.core.clip import Clip
from aqara_video.core.video_reader import Format, VideoMetadata, VideoStream


@pytest.fixture
def mock_stream() -> Mock:
    """Fixture to create a mock video stream object."""
    mock_stream = Mock(spec=VideoStream)
    mock_stream.width = 1920
    mock_stream.height = 1080
    mock_stream.codec_type = "video"
    mock_stream.avg_frame_rate = "30/1"
    return mock_stream


@pytest.fixture
def mock_format() -> Mock:
    """Fixture to create a mock format object."""
    mock_format = Mock(spec=Format)
    mock_format.duration = "10.0"
    return mock_format


@pytest.fixture
def mock_video_reader(
    monkeypatch: MonkeyPatch, mock_stream: Mock, mock_format: Mock
) -> Mock:
    """Fixture to mock the VideoReader class and its instance.

    Returns a single enhanced mock object with both class and instance capabilities:
    - Use mock_video_reader.assert_called_once_with() to verify class instantiation
    - Use mock_video_reader.instance to access the reader instance methods
    """
    # Create mock reader instance
    mock_reader_instance = Mock()
    mock_reader_instance.metadata = VideoMetadata(
        streams=[mock_stream], format=mock_format
    )
    mock_reader_instance.fps = 30.0
    mock_reader_instance.width = 1920
    mock_reader_instance.height = 1080
    mock_reader_instance.read_frame.return_value = "mock_image"
    mock_frames = [(i, f"frame_{i}") for i in range(5)]
    mock_reader_instance.frames.return_value = iter(mock_frames)

    # Create an enhanced mock for the VideoReader class
    mock_reader = Mock(return_value=mock_reader_instance)

    # Add the instance as an attribute of the mock class
    mock_reader.instance = mock_reader_instance

    # Patch the VideoReader import in the clip module
    monkeypatch.setattr("aqara_video.core.clip.VideoReader", mock_reader)

    return mock_reader


@pytest.fixture
def clip_data() -> Dict[str, Any]:
    """Fixture providing common test data for clips."""
    return {
        "camera_id": "camera1",
        "path": Path("/test/video.mp4"),
        "timestamp": datetime(2023, 1, 1, 12, 0, 0),
    }


def test_clip_initialization_and_attributes(
    mock_video_reader: Mock, clip_data: Dict[str, Any]
) -> None:
    """Test that the Clip is initialized with the correct attributes and string representation."""
    clip = Clip(
        camera_id=clip_data["camera_id"],
        path=clip_data["path"],
        timestamp=clip_data["timestamp"],
    )

    # Verify VideoReader was instantiated with the correct path
    mock_video_reader.assert_called_once_with(clip_data["path"])

    # Check instance attributes
    assert clip.camera_id == clip_data["camera_id"]
    assert clip.path == clip_data["path"]
    assert clip.timestamp == clip_data["timestamp"]

    # Test string representation
    expected_str = f"{clip_data['timestamp']} - {clip_data['path']}"
    assert str(clip) == expected_str

    # Test immutability
    with pytest.raises(AttributeError):
        clip.camera_id = "new_camera"


def test_video_properties(mock_video_reader: Mock, clip_data: Dict[str, Any]) -> None:
    """Test that the Clip correctly forwards properties from VideoReader."""
    clip = Clip(
        camera_id=clip_data["camera_id"],
        path=clip_data["path"],
        timestamp=clip_data["timestamp"],
    )

    # Test all properties
    assert clip.metadata == mock_video_reader.instance.metadata
    assert clip.fps == 30.0
    assert clip.width == 1920
    assert clip.height == 1080


def test_video_frame_operations(
    mock_video_reader: Mock, clip_data: Dict[str, Any]
) -> None:
    """Test the frame reading and iteration methods."""
    clip = Clip(
        camera_id=clip_data["camera_id"],
        path=clip_data["path"],
        timestamp=clip_data["timestamp"],
    )

    # Test single frame reading
    frame = clip.read_frame()
    mock_video_reader.instance.read_frame.assert_called_once()
    assert frame == "mock_image"

    # Test frame iteration
    frames = list(clip.frames(buffer_size=2))
    mock_video_reader.instance.frames.assert_called_once_with(buffer_size=2)
    assert len(frames) == 5
    assert frames[0] == (0, "frame_0")
    assert frames[4] == (4, "frame_4")
