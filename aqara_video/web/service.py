import logging
from datetime import datetime
from pathlib import Path

from aqara_video.web.models import CameraInfo, ScanResult, SeekResult, VideoSegment

logger = logging.getLogger(__name__)


class Service:
    def __init__(
        self,
        root_dir: Path,
        scan_result: ScanResult,
        camera_map: dict[str, str] | None = None,
    ):
        self._root_dir = root_dir
        self._scan_result = scan_result
        self._camera_map = camera_map or {}

    def list_cameras(self) -> list[CameraInfo]:
        def map_camera(camera_id: str) -> CameraInfo:
            name = self._camera_map.get(camera_id, camera_id)
            return CameraInfo(id=camera_id, name=name)

        cameras = [
            map_camera(camera_id) for camera_id in self._scan_result.camera.keys()
        ]
        return cameras

    def list_videos(self, camera_id: str) -> list[VideoSegment]:
        camera = self._scan_result.camera[camera_id]
        return camera.segments

    def get_video_path(self, relative_path: str) -> Path:
        full_path = self._root_dir / relative_path
        if not full_path.resolve().is_relative_to(self._root_dir.resolve()):
            raise ValueError("Invalid video path")
        return full_path

    def seek(self, camera_id: str, target: datetime) -> SeekResult | None:
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

        camera = self._scan_result.camera[camera_id]
        for segment in camera.segments:
            start = datetime.fromisoformat(segment.start)
            end = datetime.fromisoformat(segment.end)
            if start <= target < end:
                offset = (target - start).total_seconds()
                return SeekResult(
                    path=segment.path,
                    offset=offset,
                )

        return None
