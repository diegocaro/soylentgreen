"""Module for video reading using ffmpeg."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import ffmpeg
import numpy as np

from .types import Image


class Stream:
    pass


class AudioStream(Stream):
    pass


@dataclass(frozen=True)
class VideoStream(Stream):
    """Represents a video stream from ffprobe metadata."""

    index: int
    codec_name: str
    codec_long_name: str
    profile: str
    codec_type: str
    codec_tag_string: str
    codec_tag: str
    width: int
    height: int
    coded_width: int
    coded_height: int
    closed_captions: int
    film_grain: int
    has_b_frames: int
    pix_fmt: str
    level: int
    chroma_location: str
    field_order: str
    refs: int
    is_avc: str
    nal_length_size: str
    id: str
    r_frame_rate: str
    avg_frame_rate: str
    time_base: str
    start_pts: int
    start_time: str
    duration_ts: int
    duration: str
    bit_rate: str
    bits_per_raw_sample: str
    nb_frames: str
    extradata_size: int
    disposition: Dict[str, int]
    tags: Dict[str, str]

    def __post_init__(self) -> None:
        assert (
            self.codec_type == "video"
        ), f"Expected codec_type 'video', got {self.codec_type}"


@dataclass
class Format:
    """Represents format information from ffprobe metadata."""

    filename: str
    nb_streams: int
    nb_programs: int
    nb_stream_groups: int
    format_name: str
    format_long_name: str
    start_time: str
    duration: str
    size: str
    bit_rate: str
    probe_score: int
    tags: Dict[str, str]


@dataclass
class VideoMetadata:
    """Represents the complete ffprobe metadata for a video file."""

    streams: List[VideoStream]
    format: Format

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMetadata":
        """Create a VideoMetadata instance from a dictionary."""
        streams = [
            VideoStream(**stream)
            for stream in data.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
        format_data = Format(**data.get("format", {}))
        return cls(streams=streams, format=format_data)

    def get_best_stream(self) -> VideoStream:
        """Returns the video stream with the highest resolution."""
        video_streams = [s for s in self.streams if s.codec_type == "video"]
        if not video_streams:
            raise ValueError("No video stream found in metadata.")
        return max(video_streams, key=lambda s: s.width * s.height)


class VideoReader:
    """
    Video reader class using ffmpeg for decoding.

    This class is designed to work with video formats that OpenCV might
    have issues with, particularly those from Aqara security cameras.
    """

    def __init__(self, path: Path):
        """
        Initialize the FFmpegReader with a clip.

        Args:
            clip: Either a Clip object or a path to a video file
        """
        self.path = path
        self._metadata: Optional[VideoMetadata] = None

    @property
    def metadata(self) -> VideoMetadata:
        """
        Get video metadata using ffprobe.

        Returns:
            A dictionary containing video metadata
        """
        if self._metadata is None:
            self._metadata = VideoMetadata.from_dict(ffmpeg.probe(str(self.path)))  # type: ignore
        return self._metadata

    @property
    def best_stream(self) -> VideoStream:
        """
        Get the best video stream from the metadata.

        Returns:
            A VideoStream object representing the best video stream
        """
        return self.metadata.get_best_stream()

    @property
    def width(self) -> int:
        """Get video width."""
        return self.best_stream.width

    @property
    def height(self) -> int:
        """Get video height."""
        return self.best_stream.height

    @property
    def fps(self) -> float:
        """Get video frames per second."""
        fps_str = self.best_stream.avg_frame_rate
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            return num / den if den else 0
        return float(fps_str)

    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        return float(self.metadata.format.duration)

    @property
    def frame_count(self) -> int:
        """
        Get total number of frames in the video.

        Calculates based on duration and FPS if not available directly.
        """
        return int(self.best_stream.nb_frames)

    def read_frame(
        self, stream_index: Optional[int] = None, do_async: bool = False
    ) -> Image:
        """
        Read the first frame from the video using synchronous ffmpeg.

        Returns:
            A numpy array containing the frame data in BGR format
        """
        stream = self.best_stream
        if stream_index is None:
            stream = self.metadata.streams[stream.index]

        if do_async:
            frame = self._read_frame_async(stream)
        else:
            frame = self._read_frame_sync(stream)
        return frame

    def _bytes_to_frame(self, in_bytes: bytes, stream: VideoStream) -> Image:
        """Convert bytes to a numpy array representing the frame."""
        return np.frombuffer(in_bytes, np.uint8).reshape(stream.height, stream.width, 3)

    def _read_frame_sync(self, stream: VideoStream) -> Image:
        out, _ = (  # type: ignore
            ffmpeg.input(str(self.path))  # type: ignore
            .output("pipe:", format="rawvideo", pix_fmt="bgr24", vframes=1)
            .global_args("-map", f"0:{stream.index}")
            .run(capture_stdout=True, quiet=True)
        )
        frame = self._bytes_to_frame(out, stream)

        return frame

    def _read_frame_async(self, stream: VideoStream) -> Image:
        # Note: this is not really async as we are waiting for the process to finish
        # process = (  # type: ignore
        #     ffmpeg.input(str(self.path))  # type: ignore
        #     .output(
        #         "pipe:", format="rawvideo", pix_fmt="bgr24", vframes=1, loglevel="quiet"
        #     )
        #     .global_args("-map", f"0:{stream.index}")
        #     .run_async(pipe_stdout=True, quiet=True)
        # )
        # in_bytes = process.stdout.read(self.width * self.height * 3)
        # frame = self._byte_to_frame(in_bytes, stream)
        # process.stdout.close()
        # process.wait()
        # return frame
        raise NotImplementedError(
            "Async reading is not implemented yet. Use synchronous reading instead."
        )

    def frames(
        self, stream_index: Optional[int] = None, buffer_size: int = 1
    ) -> Iterator[Tuple[int, Image]]:
        """
        Generate frames from video with improved memory management.

        Args:
            buffer_size: Maximum number of frames to buffer

        Returns:
            Iterator yielding (frame_id, frame) tuples
        """

        if buffer_size > 1:
            raise NotImplementedError("Buffering not implemented yet.")

        stream = self.best_stream
        if stream_index is None:
            stream = self.metadata.streams[stream.index]

        process = (
            ffmpeg.input(str(self.path))
            .output("pipe:", format="rawvideo", pix_fmt="bgr24", loglevel="quiet")
            .global_args("-map", f"0:{stream.index}")
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

                frame = self._bytes_to_frame(in_bytes, stream)
                yield (frame_id, frame)

        finally:
            # Ensure resources are properly cleaned up
            process.stdout.close()
            process.wait()

    # def read_frame_at_time(self, time_sec: float) -> np.ndarray:
    #     """
    #     Read a specific frame at the given time position.

    #     Args:
    #         time_sec: Time position in seconds

    #     Returns:
    #         A numpy array containing the frame data in RGB format
    #     """
    #     if time_sec < 0 or time_sec > self.duration:
    #         raise ValueError(
    #             f"Time {time_sec} is outside video duration {self.duration}"
    #         )

    #     out, _ = (
    #         ffmpeg.input(str(self.path), ss=time_sec)
    #         .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1)
    #         .run(capture_stdout=True, quiet=True)
    #     )

    #     # Convert to numpy array
    #     frame = np.frombuffer(out, np.uint8).reshape(self.height, self.width, 3)
    #     return frame

    # def read_frames(
    #     self, start_time: Optional[float] = None, end_time: Optional[float] = None
    # ) -> Iterator[np.ndarray]:
    #     """
    #     Read frames between start_time and end_time.

    #     Args:
    #         start_time: Start time in seconds (default: 0)
    #         end_time: End time in seconds (default: end of video)

    #     Yields:
    #         Numpy arrays containing frame data in RGB format
    #     """
    #     start_time = start_time or 0
    #     end_time = end_time or self.duration

    #     if start_time < 0 or start_time >= self.duration:
    #         raise ValueError(f"Start time {start_time} is outside video duration")

    #     if end_time <= start_time or end_time > self.duration:
    #         end_time = self.duration

    #     # Create ffmpeg input with time range
    #     input_args = {"ss": start_time}
    #     if end_time < self.duration:
    #         input_args["t"] = end_time - start_time

    #     # Set up ffmpeg process
    #     process = (
    #         ffmpeg.input(str(self.path), **input_args)
    #         .output("pipe:", format="rawvideo", pix_fmt="rgb24")
    #         .run_async(pipe_stdout=True, quiet=True)
    #     )

    #     # Read frames from process
    #     frame_size = self.width * self.height * 3  # RGB = 3 bytes per pixel
    #     while True:
    #         in_bytes = process.stdout.read(frame_size)
    #         if not in_bytes:
    #             break

    #         if len(in_bytes) != frame_size:
    #             # Partial frame, discard
    #             break

    #         # Convert to numpy array
    #         frame = np.frombuffer(in_bytes, np.uint8).reshape(
    #             self.height, self.width, 3
    #         )
    #         yield frame

    #     process.stdout.close()
    #     process.wait()

    # def extract_frame(
    #     self, output_path: Union[str, Path], time_sec: float, format: str = "jpg"
    # ) -> Path:
    #     """
    #     Extract a single frame to an image file.

    #     Args:
    #         output_path: Path to save the extracted frame
    #         time_sec: Time position in seconds
    #         format: Output image format (jpg, png, etc.)

    #     Returns:
    #         Path to the saved image
    #     """
    #     output_path = Path(output_path)
    #     if not output_path.parent.exists():
    #         output_path.parent.mkdir(parents=True)

    #     # Ensure correct extension
    #     if not str(output_path).lower().endswith(f".{format.lower()}"):
    #         output_path = output_path.with_suffix(f".{format.lower()}")

    #     # Extract frame using ffmpeg
    #     (
    #         ffmpeg.input(str(self.path), ss=time_sec)
    #         .output(str(output_path), vframes=1)
    #         .overwrite_output()
    #         .run(quiet=True)
    #     )

    #     return output_path

    # def get_frame_batch(
    #     self, times: list[float], output_shape: Optional[Tuple[int, int]] = None
    # ) -> np.ndarray:
    #     """
    #     Get multiple frames at specified time positions.

    #     Useful for preparing batches for PyTorch models.

    #     Args:
    #         times: List of time positions in seconds
    #         output_shape: Optional (height, width) to resize frames

    #     Returns:
    #         Numpy array of shape (len(times), height, width, 3)
    #     """
    #     frames = []
    #     for time_sec in times:
    #         frame = self.read_frame_at_time(time_sec)

    #         # Resize if needed
    #         if output_shape is not None:
    #             # This is a simple resize implementation
    #             # For production, consider using a proper resize function
    #             h, w = output_shape
    #             temp_file = "temp_resize.jpg"

    #             # Use ffmpeg for high-quality resizing
    #             (
    #                 ffmpeg.input(
    #                     "pipe:",
    #                     format="rawvideo",
    #                     pix_fmt="rgb24",
    #                     s=f"{self.width}x{self.height}",
    #                 )
    #                 .output(temp_file, vframes=1, s=f"{w}x{h}")
    #                 .overwrite_output()
    #                 .run(input=frame.tobytes(), quiet=True)
    #             )

    #             # Read back the resized frame
    #             resize_out, _ = (
    #                 ffmpeg.input(temp_file)
    #                 .output("pipe:", format="rawvideo", pix_fmt="rgb24")
    #                 .run(capture_stdout=True, quiet=True)
    #             )

    #             frame = np.frombuffer(resize_out, np.uint8).reshape(h, w, 3)

    #             # Clean up temporary file
    #             import os

    #             os.remove(temp_file)

    #         frames.append(frame)

    #     return np.array(frames)
