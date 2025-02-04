import cv2
import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def area(self) -> int:
        return self.width * self.height


COMMON_RESOLUTIONS: List[Resolution] = [
    Resolution(320, 240),
    Resolution(640, 480),
    Resolution(800, 600),
    Resolution(1024, 768),
    Resolution(1280, 720),
    Resolution(1920, 1080),
    Resolution(2560, 1440),
    Resolution(3840, 2160),
]

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class VideoCapture:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def get_streams(self) -> List[Resolution]:
        streams = []
        for i in range(10):  # OpenCV typically supports up to 10 streams
            self.cap.set(cv2.CAP_PROP_VIDEO_STREAM, i)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width and height:
                streams.append(Resolution(width, height))
        return streams

    def get_supported_resolutions(self) -> List[Resolution]:
        available = []
        original = Resolution(
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        for res in COMMON_RESOLUTIONS:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res.height)
            actual = Resolution(
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            if actual == res:
                available.append(res)

        # Restore original resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original.height)
        return available

    def capture_frame(
        self,
        frame_number: Optional[int] = None,
        stream_index: Optional[int] = None,
        resolution: Optional[Resolution] = None,
    ) -> Optional[cv2.Mat]:
        if resolution and isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution.height)

        if stream_index is not None and isinstance(self.source, str):
            self.cap.set(cv2.CAP_PROP_VIDEO_STREAM, stream_index)

        if isinstance(self.source, str) and frame_number is not None:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_number >= total_frames:
                raise ValueError(
                    f"Frame {frame_number} out of range. Video has {total_frames} frames."
                )
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        return frame if ret else None


def setup_logging():
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, datefmt=LOG_DATE_FORMAT)
    return logging.getLogger(__name__)


def parse_resolution(res_str: str) -> Optional[Resolution]:
    try:
        width, height = map(int, res_str.split("x"))
        return Resolution(width, height)
    except ValueError:
        return None


def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Capture a frame from video source")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source (0 for webcam, or path to video file)",
    )
    parser.add_argument(
        "--frame", type=int, help="Frame number to capture (only for video files)"
    )
    parser.add_argument("--output", default="captured_frame.jpg", help="Output path")
    parser.add_argument("--stream", type=int, help="Stream index to capture")
    parser.add_argument("--resolution", help="Desired resolution (e.g., 1920x1080)")

    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    requested_resolution = (
        parse_resolution(args.resolution) if args.resolution else None
    )

    try:
        with VideoCapture(source) as vc:
            if isinstance(source, int):
                resolutions = vc.get_supported_resolutions()
                logger.info(
                    "Available resolutions: %s",
                    ", ".join([str(r) for r in resolutions]),
                )
                resolution = (
                    requested_resolution
                    if requested_resolution in resolutions
                    else max(resolutions, key=lambda r: r.area)
                )
                logger.info(f"Using resolution: {resolution}")
            else:
                resolution = None
                streams = vc.get_streams()
                logger.info(
                    "Available streams: %s",
                    ", ".join([f"{i}:{r}" for i, r in enumerate(streams)]),
                )
                stream_index = (
                    args.stream
                    if args.stream is not None
                    else streams.index(max(streams, key=lambda r: r.area))
                )
                logger.info(f"Using stream {stream_index}")

            frame = vc.capture_frame(args.frame, args.stream, resolution)
            if frame is not None:
                cv2.imwrite(args.output, frame)
                logger.info(f"Frame saved to {args.output}")
            else:
                logger.error("Failed to capture frame")

    except ValueError as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()
