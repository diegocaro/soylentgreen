import logging
from pathlib import Path

from aqara_video.core.clip import Clip
from aqara_video.core.factory import TimelineFactory
from aqara_video.providers.aqara import AqaraProvider
from aqara_video.web.config import CLIP_DURATION, SCAN_RESULT_FILE, VIDEO_DIR
from aqara_video.web.models import ScanResult, VideoList, VideoSegment

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
                start=start.isoformat(),
                end=end.isoformat(),
                path=str(clip.path.relative_to(self._root_dir)),
            )

        segments = [transform(clip) for clip in timeline.clips]
        return VideoList(segments=segments)


if __name__ == "__main__":
    manager = ScanManager(VIDEO_DIR)
    cameras = manager.scan_cameras()
    logger.info("Cameras found:")
    for camera in cameras:
        logger.info(f" - {camera}")

    scan = {camera: manager.scan_videos(camera) for camera in cameras}
    result = ScanResult(camera=scan)

    with open(SCAN_RESULT_FILE, "w") as f:
        f.write(result.model_dump_json(indent=2))
    logger.info(f"Scan result saved to {SCAN_RESULT_FILE}")
