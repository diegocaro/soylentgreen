import logging
from datetime import datetime, timedelta
from pathlib import Path

from api.models import (
    CameraInfo,
    CameraLabels,
    LabelsByCamera,
    ScanResult,
    SeekResult,
    TimeInterval,
    VideoSegment,
)

logger = logging.getLogger(__name__)


class Service:
    def __init__(
        self,
        root_dir: Path,
        scan_result: ScanResult,
        labels_timeline: LabelsByCamera,
        camera_map: dict[str, str] | None = None,
    ):
        self._root_dir = root_dir
        self._scan_result = scan_result
        self._labels_timeline = labels_timeline
        self._camera_map = camera_map or {}

    def list_cameras(self) -> list[CameraInfo]:
        def map_camera(camera_id: str) -> CameraInfo:
            name = self._camera_map.get(camera_id, camera_id)
            return CameraInfo(id=camera_id, name=name)

        cameras = [
            map_camera(camera_id) for camera_id in self._scan_result.cameras.keys()
        ]
        return cameras

    def list_videos(self, camera_id: str) -> list[VideoSegment]:
        camera = self._scan_result.cameras[camera_id]
        return camera.segments

    def get_video_path(self, relative_path: str) -> Path:
        full_path = self._root_dir / relative_path
        if not full_path.resolve().is_relative_to(self._root_dir.resolve()):
            raise ValueError("Invalid video path")
        return full_path

    def _search(
        self, segments: list[VideoSegment], target: datetime, return_next: bool = False
    ) -> VideoSegment | None:

        # assert segments are sorted by start time
        assert all(
            segments[i].start <= segments[i + 1].start for i in range(len(segments) - 1)
        )

        found_at = None
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            if start <= target < end:
                found_at = i
                break

        if found_at is not None:
            index = (
                found_at + 1
                if return_next and found_at + 1 < len(segments)
                else found_at
            )
            return segments[index]
        return None

    def seek(
        self, camera_id: str, target: datetime, return_next: bool = False
    ) -> SeekResult | None:
        """
        Given an absolute time (ISO string), find which clip covers it and the offset in seconds.
        """
        # timeline = TimelineFactory.create_timeline(self._root_dir / camera_id)
        # for clip in timeline.clips:
        #     start = clip.timestamp
        #     end = start + CLIP_DURATION
        #     if start <= target < end:
        #         offset = (target - start).total_seconds()
        #         return SeekResult(
        #             path=str(clip.path.relative_to(self._root_dir)),
        #             offset=offset,
        #         )
        # TODO: Preprocess the datetime fromisoformat

        cameras = self._scan_result.cameras[camera_id]

        result = self._search(cameras.segments, target, return_next=return_next)
        # rounded_search = False
        # if not result:
        #     # search with the target rounded to the next second
        #     logger.info("No exact match found, trying rounded search")
        #     rounded_search = True
        #     target_rounded = target.replace(microsecond=0) + timedelta(seconds=1)
        #     result = self._search(
        #         cameras.segments, target_rounded
        #     )

        if result:
            start = result.start
            offset = (target - start).total_seconds()
            # if rounded_search or return_next:
            if return_next:
                offset = 0
            return SeekResult(
                path=result.path,
                offset=offset,
            )

        return None

    def get_labels_timeline(self, camera_id: str) -> CameraLabels:
        return self._labels_timeline.cameras.get(camera_id, CameraLabels(labels={}))

    def list_intervals(
        self, camera_id: str, max_gap_seconds: int = 30
    ) -> list[TimeInterval]:
        """
        Get merged time intervals for a camera where video clips exist.
        Segments at most max_gap_seconds apart are merged into one interval.
        """
        camera = self._scan_result.cameras[camera_id]
        segments = sorted(camera.segments, key=lambda s: s.start)

        if not segments:
            return []

        merged_intervals = []
        current_start = segments[0].start
        current_end = segments[0].end

        for segment in segments[1:]:
            gap = (segment.start - current_end).total_seconds()
            if gap <= max_gap_seconds:
                current_end = max(current_end, segment.end)
            else:
                merged_intervals.append(
                    TimeInterval(start=current_start, end=current_end)
                )
                current_start = segment.start
                current_end = segment.end

        merged_intervals.append(TimeInterval(start=current_start, end=current_end))
        return merged_intervals
