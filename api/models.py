from datetime import datetime

from pydantic import BaseModel


class CameraInfo(BaseModel):
    id: str
    name: str


class VideoSegment(BaseModel):
    name: str
    start: datetime
    end: datetime
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


class IntervalTimestamp(BaseModel):
    start: datetime
    end: datetime

    # @field_validator("end")
    # def end_after_start(cls, v, info):
    #     if "start" in info.data and v <= info.data["start"]:
    #         raise ValueError("end must be after start")
    #     return v


class LabelTimeline(BaseModel):
    intervals: list[IntervalTimestamp]


class CameraLabels(BaseModel):
    labels: dict[str, LabelTimeline]


class LabelsByCamera(BaseModel):
    cameras: dict[str, CameraLabels]
