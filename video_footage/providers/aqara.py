import logging
import os
from datetime import datetime
from functools import cached_property
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..core.clip import Clip
from ..core.provider import CameraProvider

logger = logging.getLogger(__name__)


AQARA_PREFIX_DIR = "lumi1."


class AqaraProvider(CameraProvider):
    """Provider for Aqara cameras."""

    def validate_directory(self, path: Path) -> bool:
        """Check if the directory is a valid Aqara camera directory."""
        return self._is_aqara_path(path)

    @classmethod
    def _is_aqara_path(cls, path: Path) -> bool:
        return path.is_dir() and path.name.startswith(AQARA_PREFIX_DIR)

    def extract_camera_id(self, path: Path) -> str:
        """Extract camera ID from the path."""
        return path.name

    def load_clips(self, path: Path) -> list[Clip]:
        """Load clips from the specified path using Aqara's format."""
        clips = []
        files = list(path.glob("*/*.mp4"))
        logger.info(f"Loading clips from {path}, found {len(files)} files")
        for file in files:
            try:
                clip = self.create_clip(file)
                clips.append(clip)
            except ValueError as e:
                logger.warning(f"Skipping file {file}: {e}")
                continue
        return sorted(clips, key=lambda clip: clip.timestamp)

    def create_clip(self, path: Path) -> Clip:
        """Create an Aqara clip from a file path."""
        timestamp = self.extract_timestamp(path)
        camera_id = self._extract_camera_id_from_path(path)
        return Clip(camera_id=camera_id, path=path, timestamp=timestamp)

    @cached_property
    def timezone(self) -> ZoneInfo:
        # Read timezone from environment variable AQARA_TIMEZONE, default to UTC
        _tz_name = os.environ.get("AQARA_TIMEZONE", "UTC")
        try:
            zone = ZoneInfo(_tz_name)
        except ZoneInfoNotFoundError:
            logger.warning(f"Invalid timezone '{_tz_name}', defaulting to UTC.")
            zone = ZoneInfo("UTC")
        return zone

    def extract_timestamp(self, path: Path) -> datetime:
        """Extract timestamp from Aqara path format."""
        # lumi1.54ef44457bc9/20250207/082900.mp4 -> 2025-02-07 08:29:00 in configured timezone
        hhmmss = path.stem
        yyyymmdd = path.parent.stem
        try:
            naive_dt = datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
        except ValueError as e:
            raise ValueError("Invalid Aqara clip path format") from e
        localized_dt = naive_dt.replace(tzinfo=self.timezone)
        return localized_dt

    def _extract_camera_id_from_path(self, path: Path) -> str:
        """Extract camera ID from Aqara path format."""
        # lumi1.54ef44457bc9/20250207/082900.mp4 -> lumi1.54ef44457bc9
        return path.parts[-3]

    @classmethod
    def cameras_in_dir(cls, root_dir: Path) -> list[str]:
        """
        Get a list of all camera IDs found in the root directory.
        Scans the root directory for subdirectories that represent camera IDs.

        Args:
            root_dir: Path to the root directory containing camera folders

        Returns:
            A list of unique camera ID strings
        """
        camera_dirs = [d.name for d in root_dir.iterdir() if cls._is_aqara_path(d)]

        return sorted(set(camera_dirs))
