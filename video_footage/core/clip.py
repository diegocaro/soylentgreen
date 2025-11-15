from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator

from .types import ImageCV
from .video_reader import VideoFrame, VideoMetadata, VideoReader


@dataclass(frozen=True)
class Clip:
    camera_id: str
    path: Path
    timestamp: datetime
    _reader: VideoReader = field(init=False, repr=False)

    def __post_init__(self):
        # We still need __post_init__ because the path needs to be available first
        object.__setattr__(self, "_reader", VideoReader(self.path))

    def __str__(self) -> str:
        return f"{self.timestamp} - {self.path}"

    @property
    def metadata(self) -> VideoMetadata:
        return self._reader.metadata

    @property
    def fps(self) -> float:
        return self._reader.fps

    @property
    def width(self) -> int:
        return self._reader.width

    @property
    def height(self) -> int:
        return self._reader.height

    def read_frame(self) -> ImageCV:
        return self._reader.read_frame()

    def frames(self, buffer_size: int = 1) -> Iterator[VideoFrame]:
        """
        Generate frames from video clip.

        Args:
            buffer_size: Maximum number of frames to buffer

        Returns:
            Iterator yielding Frame objects
        """
        return self._reader.frames(buffer_size=buffer_size)
