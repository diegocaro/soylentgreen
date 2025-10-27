from pydantic import BaseModel, Field


class CameraInfo(BaseModel):
    id: str
    name: str


class LabeledIntervalTimestamp(BaseModel):
    label: str
    start: str  # ISO formatted datetime string
    end: str  # ISO formatted datetime string


class VideoSegment(BaseModel):
    name: str
    start: str  # ISO formatted datetime string
    end: str  # ISO formatted datetime string
    path: str  # Relative path to the video file
    intervals: list[LabeledIntervalTimestamp] = Field(default_factory=list)


class VideoList(BaseModel):
    segments: list[VideoSegment]


class SeekResult(BaseModel):
    path: str  # Relative path to the video file
    offset: float  # Offset in seconds within the video file


class ScanResult(BaseModel):
    camera: dict[str, VideoList]


class LabeledInterval(BaseModel):
    label: str
    start: float  # Start time in seconds within the video
    end: float  # End time in seconds within the video


class VideoDetectionSummary(BaseModel):
    detections: dict[str, list[LabeledInterval]]  # key: video segment path
