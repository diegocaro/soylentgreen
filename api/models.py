from datetime import datetime
from collections.abc import Sequence

from pydantic import BaseModel


class CameraInfo(BaseModel):
    id: str
    name: str


class TimeInterval(BaseModel):
    start: datetime
    end: datetime

    @classmethod
    def merge_intervals(
        cls, intervals: Sequence["TimeInterval"], max_gap_seconds: int = 30
    ) -> list["TimeInterval"]:
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        merged_intervals = []
        current_start = sorted_intervals[0].start
        current_end = sorted_intervals[0].end

        for interval in sorted_intervals[1:]:
            gap = (interval.start - current_end).total_seconds()
            if gap <= max_gap_seconds:
                current_end = max(current_end, interval.end)
            else:
                merged_intervals.append(cls(start=current_start, end=current_end))
                current_start = interval.start
                current_end = interval.end

        merged_intervals.append(cls(start=current_start, end=current_end))
        return merged_intervals


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
