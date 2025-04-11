from pathlib import Path

import ffmpeg

import aqara_video.core.constants as c
from aqara_video.core.types import Image


class VideoWriter:
    """
    A class to handle video writing operations using ffmpeg.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        framerate: int = 30,
        pix_fmt_input: str = c.PIXEL_FORMAT,
        crf: int = c.CRF_VALUE,
        vcodec: str = "libx264",
    ):
        """
        Initialize a VideoWriter for creating videos.

        Args:
            output_path: Path to the output video file
            frame_size: Tuple of (width, height)
            framerate: Frames per second for the output video
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.framerate = framerate
        self.pix_fmt_input = pix_fmt_input
        self.crf = crf
        self.vcodec = vcodec
        self.process = None

    def open(self) -> None:
        """
        Open the video writer and prepare it for writing frames.
        """
        self.process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=self.pix_fmt_input,
                s=f"{self.width}x{self.height}",
                framerate=self.framerate,
            )
            .output(
                str(self.output_path),
                pix_fmt="yuv420p",
                vcodec=self.vcodec,
                vf="format=yuv420p",
                crf=self.crf,  # Constant Rate Factor or quality
                loglevel="quiet",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write_frame(self, frame: Image) -> None:
        """
        Write a frame to the video.

        Args:
            frame: Numpy array with shape (height, width, 3)
        """
        if frame is None:
            return

        if self.process is None:
            raise RuntimeError("VideoWriter not opened. Call open() first.")

        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        """
        Close the video writer and finalize the video file.
        """
        if self.process is None:
            return

        self.process.stdin.close()
        self.process.wait()
        self.process = None
