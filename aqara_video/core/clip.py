from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import ffmpeg
import numpy as np

from .types import Image


@dataclass
class ClipMetadata:
    width: int
    height: int
    stream_index: int
    fps: float

    @classmethod
    def from_metadata(cls, stream: Dict[str, Any]) -> "ClipMetadata":
        return cls(
            width=int(stream["width"]),
            height=int(stream["height"]),
            stream_index=int(stream["index"]),
            fps=float(stream["avg_frame_rate"].split("/")[0])
            / float(stream["avg_frame_rate"].split("/")[1]),
        )


@dataclass(frozen=True)
class Clip:
    camera_id: str
    path: Path
    timestamp: datetime
    metadata: Optional[ClipMetadata] = None
    best_stream_index: Optional[int] = None

    # @classmethod
    # def _timestamp_from_path(cls, path: Path, is_utc: bool = True) -> datetime:
    #     # lumi1.54ef44457bc9/20250207/082900.mp4 -> 2025-02-07 08:29:00 UTC
    #     hhmmss = path.stem
    #     yyyymmdd = path.parent.stem
    #     timestamp = datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
    #     if is_utc:
    #         timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
    #         timestamp = timestamp_utc.astimezone()  # Convert to local timezone
    #     return timestamp

    # @classmethod
    # def _camera_id_from_path(cls, path: Path) -> str:
    #     # lumi1.54ef44457bc9/20250207/082900.mp4 -> lumi1.54ef44457bc9
    #     return path.parts[-3]

    # @classmethod
    # def from_path(cls, path: Path, metadata: Optional[ClipMetadata] = None) -> "Clip":
    #     """Factory method that handles timestamp creation"""
    #     timestamp = cls._timestamp_from_path(path)
    #     camera_id = cls._camera_id_from_path(path)
    #     return cls(
    #         camera_id=camera_id, path=path, timestamp=timestamp, metadata=metadata
    #     )

    def load_metadata(self) -> "Clip":
        """Load metadata from the video file."""

        metadata = self.get_metadata()
        return replace(self, metadata=self.best_video_stream(metadata))

    def __str__(self) -> str:
        return f"{self.timestamp} - {self.path}"

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        try:
            probe = ffmpeg.probe(str(self.path))
            return probe
        except ffmpeg.Error as e:
            print(f"Error probing {self.path}: {e}")
            return None

    def best_video_stream(self, metadata: dict) -> Optional[ClipMetadata]:
        if not metadata:
            return None
        hd_stream = max(
            (s for s in metadata["streams"] if s["codec_type"] == "video"),
            key=lambda s: s["width"],
        )
        return ClipMetadata.from_metadata(hd_stream)

    def read_frame_opencv(self) -> Optional[Image]:
        cap = cv2.VideoCapture(self.path.as_posix())
        # stream_index = self.get_hd_track()
        # print(f"Reading stream {stream_index}")

        # cap.set(cv.CAP_PROP_VIDEO_STREAM, stream_index)
        frame = None
        while cap.isOpened() and frame is None:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?).")
                break
        cap.release()
        return frame

    def read_frame_ffmpeg_sync(self):
        stream = self.stream
        if not stream:
            return None

        out, _ = (
            ffmpeg.input(str(self.path))
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt=PIXEL_FORMAT,
                vframes=1,
                loglevel="quiet",
            )
            .global_args("-map", f"0:{stream.stream_index}")
            .run()
        )

        return np.frombuffer(out, np.uint8).reshape([stream.height, stream.width, 3])

    def read_frame(self) -> Image:
        stream = self.stream
        if not stream:
            return None

        process = (
            ffmpeg.input(str(self.path))
            .output(
                "pipe:", format="rawvideo", pix_fmt="bgr24", vframes=1, loglevel="quiet"
            )
            .global_args("-map", f"0:{stream.stream_index}")
            .run_async(pipe_stdout=True)
        )

        in_bytes = process.stdout.read(stream.width * stream.height * 3)
        frame = np.frombuffer(in_bytes, np.uint8).reshape(
            [stream.height, stream.width, 3]
        )

        process.stdout.close()
        process.wait()

        return frame

    @property
    def stream(self) -> Optional[ClipMetadata]:
        return self.best_video_stream(self.get_metadata())

    def frames(self, buffer_size=10) -> Iterator[Tuple[int, Image]]:
        """
        Generate frames from video with improved memory management.

        Args:
            buffer_size: Maximum number of frames to buffer

        Returns:
            Iterator yielding (frame_id, frame) tuples
        """
        stream = self.stream
        if not stream:
            return

        process = (
            ffmpeg.input(str(self.path))
            .output("pipe:", format="rawvideo", pix_fmt="bgr24", loglevel="quiet")
            .global_args("-map", f"0:{stream.stream_index}")
            .run_async(pipe_stdout=True)
        )

        try:
            frame_id = -1
            bytes_per_frame = stream.width * stream.height * 3

            while True:
                frame_id += 1
                in_bytes = process.stdout.read(bytes_per_frame)
                if not in_bytes or len(in_bytes) < bytes_per_frame:
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape(
                    [stream.height, stream.width, 3]
                )
                yield (frame_id, frame)

        finally:
            # Ensure resources are properly cleaned up
            process.stdout.close()
            process.wait()
