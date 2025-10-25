import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from aqara_video.core.clip import Clip
from aqara_video.core.factory import TimelineFactory
from aqara_video.providers.aqara import AqaraProvider
from aqara_video.web.models import SeekResult, VideoSegment

CLIP_DURATION = timedelta(minutes=1)


logger = logging.getLogger(__name__)


class Service:
    def __init__(self, root_dir: Path):
        self._root_dir = root_dir

    def list_cameras(self) -> list[str]:
        cameras = AqaraProvider.cameras_in_dir(self._root_dir)
        return cameras

    def list_videos(self, camera_id: str) -> list[VideoSegment]:
        timeline = TimelineFactory.create_timeline(self._root_dir / camera_id)

        def transform(clip: Clip) -> VideoSegment:
            start = clip.timestamp
            end = start + CLIP_DURATION
            return VideoSegment(
                name=clip.path.name,
                start=start.isoformat(),
                end=end.isoformat(),
                path=str(clip.path.relative_to(self._root_dir)),
            )

        ans = [transform(clip) for clip in timeline.clips]
        return ans

    def get_video_path(self, relative_path: str) -> Path:
        full_path = self._root_dir / relative_path
        if not full_path.resolve().is_relative_to(self._root_dir.resolve()):
            raise ValueError("Invalid video path")
        return full_path

    def seek(self, camera_id: str, target: datetime) -> SeekResult | None:
        """
        Given an absolute time (ISO string), find which clip covers it and the offset in seconds.
        """
        timeline = TimelineFactory.create_timeline(self._root_dir / camera_id)

        for clip in timeline.clips:
            start = clip.timestamp
            end = start + CLIP_DURATION
            if start <= target < end:
                offset = (target - start).total_seconds()
                return SeekResult(
                    path=str(clip.path.relative_to(self._root_dir)),
                    offset=offset,
                )

        return None
