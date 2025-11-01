from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from .clip import Clip


class CameraProvider(ABC):
    """Base class for camera providers."""

    @abstractmethod
    def validate_directory(self, path: Path) -> bool:
        """Validate if the directory belongs to this provider."""
        pass

    @abstractmethod
    def extract_camera_id(self, path: Path) -> str:
        """Extract camera ID from the path."""
        pass

    @abstractmethod
    def load_clips(self, path: Path) -> list[Clip]:
        """Load clips from the given path."""
        pass

    @abstractmethod
    def create_clip(self, path: Path) -> Clip:
        """Create a clip from a file path."""
        pass

    @abstractmethod
    def extract_timestamp(self, path: Path) -> datetime:
        """Extract timestamp from the path according to provider's format."""
        pass
