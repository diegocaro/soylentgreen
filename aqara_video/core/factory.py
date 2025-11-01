from pathlib import Path

from ..providers.aqara import AqaraProvider
from .provider import CameraProvider
from .timeline import Timeline, TimelineError


class NoCompatibleProviderError(TimelineError):
    """Exception raised when no provider can handle the given directory."""

    def __init__(self, path: Path, available_providers: list[CameraProvider]):
        provider_names = [
            provider.__class__.__name__ for provider in available_providers
        ]
        providers_str = ", ".join(provider_names)
        self.message = (
            f"No compatible camera vendor found for directory: {path}\n"
            f"Available vendors: {providers_str}"
        )
        super().__init__(self.message)


class TimelineFactory:
    """Factory for creating timelines with the appropriate provider."""

    PROVIDERS: list[CameraProvider] = [AqaraProvider()]

    @classmethod
    def create_timeline(cls, clips_path: Path) -> Timeline:
        """Create a timeline with the appropriate provider for the given path."""

        if not clips_path.exists():
            raise TimelineError(f"Directory does not exist: {clips_path}")
        if not clips_path.is_dir():
            raise TimelineError(f"Invalid directory: {clips_path}")
        for provider in cls.PROVIDERS:
            if provider.validate_directory(clips_path):
                return Timeline(clips_path, provider)

        raise NoCompatibleProviderError(clips_path, cls.PROVIDERS)
