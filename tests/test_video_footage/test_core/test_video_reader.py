import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from video_footage.core.video_reader import (
    Format,
    VideoFrame,
    VideoMetadata,
    VideoReader,
    VideoStream,
)


@pytest.fixture
def mock_path_exist(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)


@pytest.fixture
def sample_video_metadata() -> dict[str, Any]:
    """Fixture to provide sample video metadata from JSON file."""
    json_path = Path(__file__).parent / "sample_video_metadata.json"
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture
def mock_video_stream(sample_video_metadata: dict[str, Any]) -> VideoStream:
    """Fixture to create a real VideoStream object from sample metadata."""
    video_stream_data = next(
        stream
        for stream in sample_video_metadata["streams"]
        if stream["codec_type"] == "video"
    )
    return VideoStream(**video_stream_data)


@pytest.fixture
def mock_format(sample_video_metadata: dict[str, Any]) -> Format:
    """Fixture to create a real Format object from sample metadata."""
    return Format(**sample_video_metadata["format"])


@pytest.fixture
def mock_video_metadata(
    mock_video_stream: VideoStream, mock_format: Format
) -> VideoMetadata:
    """Fixture to create a real VideoMetadata object."""
    return VideoMetadata(streams=[mock_video_stream], format=mock_format)


class MockProcess:
    """Mock class for ffmpeg process."""

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.frame_index = 0
        self.stdout = self

    def read(self, size: int) -> bytes:
        """Mock read method that returns frame bytes."""
        if self.frame_index >= len(self.frames):
            return b""
        frame = self.frames[self.frame_index]
        self.frame_index += 1
        return frame.tobytes()

    def close(self) -> None:
        """Mock close method."""
        pass

    def wait(self) -> None:
        """Mock wait method."""
        pass


def test_video_metadata_from_dict(sample_video_metadata: dict[str, Any]) -> None:
    """Test creating VideoMetadata from a dictionary."""
    metadata = VideoMetadata.from_dict(sample_video_metadata)

    assert len(metadata.streams) == 2
    assert metadata.streams[0].codec_name == "h264"
    assert metadata.streams[0].width == 640
    assert metadata.streams[0].height == 360
    assert (
        metadata.streams[1].codec_name == "h264"
    )  # Both streams are h264 in the updated JSON
    assert metadata.streams[1].width == 1920
    assert metadata.streams[1].height == 1080
    assert metadata.format.duration == "60.302000"
    assert metadata.format.bit_rate == "2527600"


def test_get_best_stream(sample_video_metadata: dict[str, Any]) -> None:
    """Test getting the best video stream."""
    metadata = VideoMetadata.from_dict(sample_video_metadata)
    best_stream = metadata.get_best_stream()

    # The second stream (index 1) has higher resolution and should be chosen
    assert best_stream.index == 1
    assert best_stream.codec_type == "video"
    assert best_stream.width == 1920
    assert best_stream.height == 1080


def test_get_best_stream_no_video_streams(
    sample_video_metadata: dict[str, Any],
) -> None:
    """Test getting the best video stream when no video streams are available."""

    format_obj = Format(**sample_video_metadata.get("format", {}))
    metadata = VideoMetadata(streams=[], format=format_obj)

    with pytest.raises(ValueError, match="No video stream found in metadata"):
        metadata.get_best_stream()


def test_video_reader_initialization(mock_path_exist):
    """Test VideoReader initialization."""
    path = Path("/path/to/test.mp4")
    reader = VideoReader(path)

    assert reader.path == path
    assert reader._metadata is None


@patch("ffmpeg.probe")
def test_video_reader_metadata(
    mock_probe: Mock, sample_video_metadata: dict[str, Any], mock_path_exist
) -> None:
    """Test the metadata property of VideoReader."""
    mock_probe.return_value = sample_video_metadata

    reader = VideoReader(Path("/path/to/test.mp4"))
    metadata = reader.metadata

    mock_probe.assert_called_once_with("/path/to/test.mp4")
    assert isinstance(metadata, VideoMetadata)
    assert len(metadata.streams) == 2
    assert metadata.format.duration == "60.302000"


@patch("ffmpeg.probe")
def test_video_reader_properties(
    mock_probe: Mock, sample_video_metadata: dict[str, Any], mock_path_exist
) -> None:
    """Test the properties of VideoReader."""
    mock_probe.return_value = sample_video_metadata

    reader = VideoReader(Path("/path/to/test.mp4"))

    # Using the values from the updated JSON file
    # The highest resolution stream (index 1) should be used
    assert reader.width == 1920
    assert reader.height == 1080
    # The fps is calculated from the avg_frame_rate of the best stream (1097000/56533 ≈ 19.4046)
    assert reader.fps == 19.404595545964305
    assert reader.duration == 60.302  # from format duration
    assert reader.frame_count == 1097  # from the best stream's nb_frames


@patch("ffmpeg.input")
@patch("ffmpeg.probe")
def test_read_frame_sync(
    mock_probe: Mock,
    mock_input: Mock,
    sample_video_metadata: dict[str, Any],
    mock_path_exist,
) -> None:
    """Test reading a frame synchronously."""
    # Mock ffmpeg.probe
    mock_probe.return_value = sample_video_metadata

    # Create a mock frame
    mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Mock the ffmpeg pipeline
    mock_output = Mock()
    mock_input.return_value.output.return_value = mock_output
    mock_output.global_args.return_value = mock_output
    mock_output.run.return_value = (mock_frame.tobytes(), None)

    reader = VideoReader(Path("/path/to/test.mp4"))
    frame = reader.read_frame()

    mock_input.assert_called_once_with("/path/to/test.mp4")
    # Updated to check for mapping to stream index 1 (the best stream based on resolution)
    mock_output.global_args.assert_called_once_with("-map", "0:1")
    mock_output.run.assert_called_once_with(capture_stdout=True, quiet=True)

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (1080, 1920, 3)


def test_read_frame_async(mock_path_exist) -> None:
    """Test that reading a frame asynchronously raises NotImplementedError."""
    reader = VideoReader(Path("/path/to/test.mp4"))

    with pytest.raises(NotImplementedError):
        reader._read_frame_async(Mock(), Mock())


@patch("ffmpeg.input")
@patch("ffmpeg.probe")
def test_frames_generator(
    mock_probe: Mock,
    mock_input: Mock,
    sample_video_metadata: dict[str, Any],
    mock_path_exist,
) -> None:
    """Test the frames generator method."""
    # Mock ffmpeg.probe
    mock_probe.return_value = sample_video_metadata

    # Create mock frames
    mock_frames = [np.ones((1080, 1920, 3), dtype=np.uint8) * i for i in range(5)]

    # Mock the ffmpeg run_async output
    mock_process = MockProcess(mock_frames)

    # Mock the ffmpeg pipeline
    mock_output = Mock()
    mock_input.return_value.output.return_value = mock_output
    mock_output.global_args.return_value = mock_output
    mock_output.run_async.return_value = mock_process

    reader = VideoReader(Path("/path/to/test.mp4"))
    frames = list(reader.frames())

    mock_input.assert_called_once_with("/path/to/test.mp4")
    # Updated to check for mapping to stream index 1 (the best stream based on resolution)
    mock_output.global_args.assert_called_once_with("-map", "0:1")
    mock_output.run_async.assert_called_once_with(pipe_stdout=True)

    assert len(frames) == 5
    for i, frame in enumerate(frames):
        assert isinstance(frame, VideoFrame)
        assert frame.frame_id == i
        assert isinstance(frame.frame, np.ndarray)
        assert frame.frame.shape == (1080, 1920, 3)
        # Check if the frame has the correct value (based on our mock implementation)
        assert np.all(frame.frame == i)


def test_frames_buffer_size_not_implemented(mock_path_exist) -> None:
    """Test that using frames with buffer_size > 1 raises NotImplementedError."""
    reader = VideoReader(Path("/path/to/test.mp4"))

    with pytest.raises(NotImplementedError):
        list(reader.frames(buffer_size=2))


def test_bytes_to_frame(mock_path_exist) -> None:
    """Test converting bytes to a frame."""
    reader = VideoReader(Path("/path/to/test.mp4"))

    # Create test data
    width, height = 4, 3
    test_stream = Mock(width=width, height=height)
    # Create a 3D array with the correct size (height, width, 3)
    test_data = np.zeros((height, width, 3), dtype=np.uint8)
    # Fill with sequential values for testing
    for h in range(height):
        for w in range(width):
            for c in range(3):
                test_data[h, w, c] = h * width * 3 + w * 3 + c

    # Convert to bytes
    test_bytes = test_data.tobytes()

    # Convert back to frame
    frame = reader._bytes_to_frame(test_bytes, test_stream)

    # Assert shape and content
    assert frame.shape == (height, width, 3)
    assert np.array_equal(frame, test_data)
