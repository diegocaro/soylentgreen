from pydantic import BaseModel


class VideoSegment(BaseModel):
    name: str
    start: str  # ISO formatted datetime string
    end: str  # ISO formatted datetime string
    path: str  # Relative path to the video file


class SeekResult(BaseModel):
    path: str  # Relative path to the video file
    offset: float  # Offset in seconds within the video file
