from datetime import date, datetime
from pathlib import Path

from .clip import Clip
from .provider import CameraProvider


class TimelineError(Exception):
    pass


class InvalidCameraDirError(TimelineError):
    def __init__(self, path: Path):
        super().__init__(f"Not a valid camera directory: {path}")
        self.path = path


# class NotAqaraCameraDirError(TimelineError):
#     def __init__(self, path: Path):
#         super().__init__(
#             f"Not a valid Aqara camera directory: {path}. Expected 'lumi1.' prefix."
#         )
#         self.path = path


class Timeline:
    """
    File structure: camera_id/day/time.mp4
    Example: lumi1.54ef44457bc9/20250207/091202.mp4
    """

    def __init__(self, clips_path: Path, provider: CameraProvider):
        self.clips_path = clips_path
        self.provider = provider
        self._clips: list[Clip] | None = None  # Lazy load clips

        self.camera_id = self.clips_path.name
        if not self.provider.validate_directory(self.clips_path):
            raise InvalidCameraDirError(self.clips_path)

    @property
    def clips(self) -> list[Clip]:
        """Return the list of clips in the timeline."""
        if self._clips is None:
            self._clips = self.provider.load_clips(self.clips_path)
        return self._clips

    # def _load_clips(self, clips_path: Path) -> List[Clip]:
    #     """Load clips from the specified path."""
    #     files = [Clip.from_path(file) for file in clips_path.glob("*/*.mp4")]
    #     ans = sorted(files, key=lambda clip: clip.timestamp)
    #     return ans

    def __str__(self) -> str:
        return "\n".join(str(clip) for clip in self.clips)

    def get_available_dates(self) -> list[date]:
        """Return a list of available dates in the timeline."""
        dates = {clip.timestamp.date() for clip in self.clips}
        return sorted(list(dates))

    def search_clips(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> list[Clip]:
        # Should improve this!!!! although the N is pretty small
        clips: list[Clip] = []
        for clip in self.clips:
            if date_from and clip.timestamp < date_from:
                continue
            if date_to and clip.timestamp > date_to:
                continue
            clips.append(clip)
        return clips

    def __len__(self) -> int:
        """Return the number of clips in the timeline."""
        return len(self.clips)
