from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    def __init__(self, base_path: Path):
        self._base_path = base_path

    def list_videos(self) -> list[dict]:
        clips = []
        for f in self._base_path.rglob("*.mp4"):
            start = extract_timestamp(f)
            if not start:
                continue
            end = start + CLIP_DURATION
            clips.append(
                {
                    "name": f.name,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "path": str(f.relative_to(self._base_path)),
                }
            )
        clips.sort(key=lambda x: x["start"])
        return clips

    def get_video_path(self, relative_path: str) -> Path:
        full_path = self._base_path / relative_path
        if not full_path.resolve().is_relative_to(self._base_path.resolve()):
            raise ValueError("Invalid video path")
        return full_path

    def seek(self, target: datetime) -> dict | None:
        """
        Given an absolute time (ISO string), find which clip covers it and the offset in seconds.
        """
        for f in self._base_path.rglob("*.mp4"):
            start = extract_timestamp(f)
            if not start:
                continue
            end = start + CLIP_DURATION
            if start <= target < end:
                offset = (target - start).total_seconds()
                return {"path": str(f.relative_to(self._base_path)), "offset": offset}
