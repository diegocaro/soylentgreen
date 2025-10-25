from datetime import datetime, timedelta, timezone
from pathlib import Path

from aqara_video.core.clip import Clip
from aqara_video.core.factory import TimelineFactory
from aqara_video.providers.aqara import AqaraProvider
from aqara_video.web.models import SeekResult, VideoSegment

CLIP_DURATION = timedelta(minutes=1)


def extract_timestamp(path: Path) -> datetime:
    """Extract timestamp from Aqara path format."""
    # lumi1.54ef44457bc9/20250207/082900.mp4 -> 2025-02-07 08:29:00 UTC
    hhmmss = path.stem
    yyyymmdd = path.parent.stem
    timestamp = datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
    timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
    return timestamp_utc.astimezone()  # Convert to local timezone


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
