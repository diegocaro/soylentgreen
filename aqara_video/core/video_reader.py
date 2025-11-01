"""Module for video reading using ffmpeg."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np

from .types import ImageCV


@dataclass(frozen=True)
class VideoFrame:
    """A video frame with its metadata."""

    frame_id: int  # frame ID
    t: float  # timestamp in seconds, not needed for now
    frame: ImageCV  # actual image data


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
    # closed_captions: int
    # film_grain: int
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
    disposition: dict[str, int]
    tags: dict[str, str]

    def __post_init__(self) -> None:
        assert self.codec_type == "video", (
            f"Expected codec_type 'video', got {self.codec_type}"
        )


@dataclass
class Format:
    """Represents format information from ffprobe metadata."""

    filename: str
    nb_streams: int
    nb_programs: int
    nb_stream_groups: int | None
    format_name: str
    format_long_name: str
    start_time: str
    duration: str
    size: str
    bit_rate: str
    probe_score: int
    tags: dict[str, str]


@dataclass
class VideoMetadata:
    """Represents the complete ffprobe metadata for a video file."""

    streams: list[VideoStream]
    format: Format

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMetadata":
        """Create a VideoMetadata instance from a dictionary."""
        streams = []
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                # Only keep fields that are in VideoStream dataclass
                allowed = {
                    k: v
                    for k, v in stream.items()
                    if k in VideoStream.__dataclass_fields__
                }
                streams.append(VideoStream(**allowed))
        format_dict = data.get("format", {})
        allowed_format = {
            k: v for k, v in format_dict.items() if k in Format.__dataclass_fields__
        }
        format_data = Format(**allowed_format)
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
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        self.path = path
        self._metadata: VideoMetadata | None = None

    @property
    def metadata(self) -> VideoMetadata:
        """
        Get video metadata using ffprobe.

        Returns:
            A dictionary containing video metadata
        """
        if self._metadata is None:
            try:
                self._metadata = VideoMetadata.from_dict(ffmpeg.probe(str(self.path)))  # type: ignore
            except ffmpeg.Error as e:
                raise RuntimeError(
                    f"Failed to probe video file {self.path}: {e.stderr.decode()}"
                ) from e
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
        self,
        time_sec: float = 0.0,
        stream_index: int | None = None,
        do_async: bool = False,
    ) -> ImageCV:
        """
        Read the first frame from the video using synchronous ffmpeg.

        Returns:
            A numpy array containing the frame data in BGR format
        """
        if time_sec < 0 or time_sec > self.duration:
            raise ValueError(
                f"Time {time_sec} is outside video duration {self.duration}"
            )

        stream = self.best_stream
        if stream_index is None:
            stream = self.metadata.streams[stream.index]

        if do_async:
            frame = self._read_frame_async(time_sec=time_sec, stream=stream)
        else:
            frame = self._read_frame_sync(time_sec=time_sec, stream=stream)
        return frame

    def _bytes_to_frame(self, in_bytes: bytes, stream: VideoStream) -> ImageCV:
        """Convert bytes to a numpy array representing the frame."""
        return np.frombuffer(in_bytes, np.uint8).reshape(stream.height, stream.width, 3)

    def _read_frame_sync(self, time_sec: float, stream: VideoStream) -> ImageCV:
        input_args = {}
        if time_sec > 0:
            input_args["ss"] = time_sec

        out, _ = (  # type: ignore
            ffmpeg.input(str(self.path), **input_args)  # type: ignore
            .output("pipe:", format="rawvideo", pix_fmt="bgr24", vframes=1)
            .global_args("-map", f"0:{stream.index}")
            .run(capture_stdout=True, quiet=True)
        )
        frame = self._bytes_to_frame(out, stream)

        return frame

    def _read_frame_async(self, time_sec: float, stream: VideoStream) -> ImageCV:
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
        self,
        stream_index: int | None = None,
        buffer_size: int = 1,
        frame_skip: int = 1,
    ) -> Iterator[VideoFrame]:
        """
        Generate frames from video with improved memory management.

        Args:
            stream_index: Optional index of the stream to use
            buffer_size: Maximum number of frames to buffer
            skip_frames: Number of frames to skip between reads

        Returns:
            Iterator yielding Frame objects
        """
        if buffer_size > 1:
            raise NotImplementedError("Buffering not implemented yet.")
        if frame_skip < 1:
            raise ValueError("skip_frames must be at least 1")

        stream = self.best_stream
        if stream_index is not None:
            stream = self.metadata.streams[stream_index]
        # TODO: add support for hwaccel if needed
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
                if frame_id % frame_skip != 0:
                    continue

                frame = self._bytes_to_frame(in_bytes, stream)
                yield VideoFrame(frame_id=frame_id, frame=frame, t=frame_id / self.fps)

        finally:
            # Ensure resources are properly cleaned up
            process.stdout.close()
            process.wait()

    def frames_with_precise_timestamp(
        self,
        start_time: float = 0.0,
        stream_index: int | None = None,
        buffer_size: int = 1,
    ) -> Iterator[VideoFrame]:
        """
        Generate frames from video with timestamps from FFmpeg.

        Args:
            start_time: Optional start time in seconds (default: 0)
            stream_index: Optional index of the stream to use
            buffer_size: Maximum number of frames to buffer

        Returns:
            Iterator yielding Frame objects with accurate timestamps
        """
        raise NotImplementedError(
            "Precise timestamp reading is not implemented yet. Use basic reading instead."
        )
        # if buffer_size > 1:
        #     raise NotImplementedError("Buffering not implemented yet.")
        # if start_time < 0 or start_time >= self.duration:
        #     raise ValueError(f"Start time {start_time} is outside video duration")
        # stream = self.best_stream
        # if stream_index is not None:
        #     stream = self.metadata.streams[stream_index]
        # input_args = {}
        # if start_time > 0:
        #     input_args["ss"] = start_time
        # process = (
        #     ffmpeg.input(str(self.path), **input_args)
        #     .filter("showinfo")
        #     .output(
        #         "pipe:",
        #         format="rawvideo",
        #         pix_fmt="bgr24",
        #         loglevel="info",  # We need at least info level to get showinfo output
        #     )
        #     .global_args("-map", f"0:{stream.index}")
        #     .run_async(pipe_stdout=True, pipe_stderr=True)
        # )
        # timestamps_queue = queue.Queue()
        # # Function to read stderr in background
        # def read_stderr():
        #     while True:
        #         line = process.stderr.readline()
        #         if not line:
        #             break
        #         decoded_line = line.decode("utf-8", errors="replace").strip()
        #         # Try to extract frame ID and timestamp from showinfo
        #         timestamp_info = _parse_showinfo_timestamp(decoded_line)
        #         if timestamp_info:
        #             timestamps_queue.put(timestamp_info)
        # def _parse_showinfo_timestamp(line: str) -> Optional[Tuple[int, float]]:
        #     """Extract frame ID and PTS time from showinfo filter output.
        #     Example line:
        #     [Parsed_showinfo_0 @ 0x7b81940] n: 132 pts: 594000 pts_time:6.6 duration: 4500...
        #     """
        #     if "showinfo" not in line:
        #         return None
        #     # Check for the standard pattern that includes frame number and timestamp
        #     pattern = r"n:\s*(\d+).+pts_time:([\d\.]+)"
        #     match = re.search(pattern, line)
        #     if match:
        #         frame_id = int(match.group(1))
        #         pts_time = float(match.group(2))
        #         return frame_id, pts_time
        #     # print("No match found in line:", line)
        #     return None
        # # Start stderr reader thread
        # stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        # stderr_thread.start()
        # try:
        #     bytes_per_frame = stream.width * stream.height * 3
        #     while True:
        #         in_bytes = process.stdout.read(bytes_per_frame)
        #         if not in_bytes or len(in_bytes) < bytes_per_frame:
        #             break
        #         frame_data = self._bytes_to_frame(in_bytes, stream)
        #         frame_id = -1
        #         frame_time = -1.0
        #         # Take just one timestamp from the queue for this frame
        #         if not timestamps_queue.empty():
        #             frame_id, frame_time = timestamps_queue.get()
        #         yield VideoFrame(
        #             frame_id=frame_id, time_sec=frame_time, frame=frame_data
        #         )
        # finally:
        #     # Ensure resources are properly cleaned up
        #     process.stdout.close()
        #     process.wait()
