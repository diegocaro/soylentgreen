from pathlib import Path
from typing import Any, Protocol


class ModelProtocol(Protocol):
    model_name: str
    model_version: str

    def predict(self, video_path: Path) -> list[tuple[float, float]]:
        """
        Run prediction on the given file path.

        Args:
            video_path (Path): Path to the input file.

        Returns:
            Any: The prediction result.
        """
        ...
