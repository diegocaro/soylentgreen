import logging
from pathlib import Path

from api.config import CLIP_DURATION
from api.models import VideoList, VideoSegment
from video_footage.core.clip import Clip
from video_footage.core.factory import TimelineFactory
from video_footage.providers.aqara import AqaraProvider

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class ScanManager:
    def __init__(self, root_dir: Path):
        self._root_dir = root_dir

    def scan_cameras(self) -> list[str]:
        cameras = AqaraProvider.cameras_in_dir(self._root_dir)
        return cameras

    def scan_videos(self, camera_id: str) -> VideoList:
        logger.info(f"Scanning videos for camera: {camera_id}")
        timeline = TimelineFactory.create_timeline(self._root_dir / camera_id)

        def transform(clip: Clip) -> VideoSegment:
            start = clip.timestamp
            end = start + CLIP_DURATION
            return VideoSegment(
                name=clip.path.name,
                start=start,
                end=end,
                path=str(clip.path.relative_to(self._root_dir)),
            )

        segments = [transform(clip) for clip in timeline.clips]
        return VideoList(segments=segments)
