from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..core.clip import Clip, ClipMetadata
from ..core.provider import CameraProvider

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

    def load_clips(self, path: Path) -> List[Clip]:
        """Load clips from the specified path using Aqara's format."""
        files = [self.create_clip(file) for file in path.glob("*/*.mp4")]
        return sorted(files, key=lambda clip: clip.timestamp)

    def create_clip(self, path: Path, metadata: Optional[ClipMetadata] = None) -> Clip:
        """Create an Aqara clip from a file path."""
        timestamp = self.extract_timestamp(path)
        camera_id = self._extract_camera_id_from_path(path)
        return Clip(
            camera_id=camera_id, path=path, timestamp=timestamp, metadata=metadata
        )

    def extract_timestamp(self, path: Path) -> datetime:
        """Extract timestamp from Aqara path format."""
        # lumi1.54ef44457bc9/20250207/082900.mp4 -> 2025-02-07 08:29:00 UTC
        hhmmss = path.stem
        yyyymmdd = path.parent.stem
        timestamp = datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
        timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
        return timestamp_utc.astimezone()  # Convert to local timezone

    def _extract_camera_id_from_path(self, path: Path) -> str:
        """Extract camera ID from Aqara path format."""
        # lumi1.54ef44457bc9/20250207/082900.mp4 -> lumi1.54ef44457bc9
        return path.parts[-3]

    @classmethod
    def cameras_in_dir(cls, root_dir: Path) -> List[str]:
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
