"""
Test script to check frame extraction and information display.
Run this script to verify that frame IDs are correctly extracted.
"""

import sys
from pathlib import Path

import pytest

from aqara_video.core.video_reader import Frame, VideoReader


@pytest.fixture
def video_path():
    return "tests/videos/living_room.mp4"


def test_frame_extraction(video_path, max_frames=200):
    """
    Test frame extraction and print frame info for diagnostic purposes.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to process
    """
    print(f"Testing frame extraction from: {video_path}")
    reader = VideoReader(Path(video_path))

    print(f"Video metadata:")
    print(f"  Resolution: {reader.width}x{reader.height}")
    print(f"  FPS: {reader.fps}")
    print(f"  Duration: {reader.duration} seconds")
    print(f"  Frame count: {reader.frame_count}")
    print("\nExtracting frames...")

    frame_count = 0
    for frame in reader.frames():
        frame_count += 1
        print(f"Frame {frame_count}: ID={frame.n}, Time={frame.time_sec:.3f}s, Shape={frame.frame.shape}")

        if frame_count >= max_frames:
            print(f"Reached max frame count of {max_frames}")
            break

    print(f"\nTotal frames processed: {frame_count}")


if __name__ == "__main__":
    # Use a test video if available, otherwise specify a path
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Make sure we're using the correct path
        filename = video_path()

    test_frame_extraction(video_path)
