from datetime import datetime

from pydantic import BaseModel


class CameraInfo(BaseModel):
    id: str
    name: str


class TimeInterval(BaseModel):
    start: datetime
    end: datetime


class VideoSegment(TimeInterval):
    name: str
    path: str  # Relative path to the video file


class VideoList(BaseModel):
    segments: list[VideoSegment]


class SeekResult(BaseModel):
    path: str  # Relative path to the video file
    offset: float  # Offset in seconds within the video file


class ScanResult(BaseModel):
    cameras: dict[str, VideoList]
    scanned_at: datetime | None = None


class LabeledInterval(BaseModel):
    label: str
    start: float  # Start time in seconds within the video
    end: float  # End time in seconds within the video


class VideoDetectionSummary(BaseModel):
    detections: dict[str, list[LabeledInterval]]  # key: video segment path


class LabelTimeline(BaseModel):
    intervals: list[TimeInterval]


class CameraLabels(BaseModel):
    labels: dict[str, LabelTimeline]


class LabelsByCamera(BaseModel):
    cameras: dict[str, CameraLabels]
