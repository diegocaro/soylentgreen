import argparse
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import ffmpeg
import cv2 as cv

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


@dataclass
class VideoStream:
    width: int
    height: int
    stream_index: int

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @classmethod
    def from_metadata(cls, stream: Dict[str, Any]) -> "VideoStream":
        return cls(
            width=int(stream["width"]),
            height=int(stream["height"]),
            stream_index=int(stream["index"]),
        )


def dynamic_range(frame: np.ndarray) -> Dict[str, float]:
    """
    Analyze the dynamic range in a frame.

    Args:
        frame: numpy array with shape (height, width, 3) in RGB format

    Returns:
        Dictionary with color distribution metrics and dynamic range info
    """
    if frame is None:
        return {
            "dynamic_range": 0.0,
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
        }

    # Calculate brightness (using standard formula: 0.2126*R + 0.7152*G + 0.0722*B)
    brightness = np.dot(frame, [0.2126, 0.7152, 0.0722])

    # Calculate dynamic range metrics
    mean_brightness = np.mean(brightness)
    std_brightness = np.std(brightness)

    # Calculate percentile-based dynamic range (5th to 95th percentile)
    p05 = np.percentile(brightness, 5)
    p95 = np.percentile(brightness, 95)
    dynamic_range = (p95 - p05) / 255.0  # Normalize to [0,1]

    distribution = {
        "dynamic_range": dynamic_range * 100,  # Convert to percentage
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
    }

    return distribution


@dataclass()
class Clip:
    path: Path
    timestamp: datetime
    metadata: Optional[dict] = None

    @staticmethod
    def timestamp_from_path(path: Path, is_utc: bool = True) -> datetime:
        # 20250207/082900.mp4 -> 2025-02-07 08:29:00 UTC
        hhmmss = path.stem
        yyyymmdd = path.parent.stem
        timestamp = datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
        if is_utc:
            timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
            timestamp = timestamp_utc.astimezone()  # Convert to local timezone
        return timestamp

    @staticmethod
    def build(path: Path) -> "Clip":
        timestamp = Clip.timestamp_from_path(path)
        return Clip(path, timestamp)

    def __str__(self) -> str:
        return f"{self.timestamp} - {self.path}"

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        try:
            probe = ffmpeg.probe(str(self.path))
            return probe
        except ffmpeg.Error as e:
            print(f"Error probing {self.path}: {e}")
            return None

    def best_video_stream(self, metadata: dict) -> Optional[VideoStream]:
        if not metadata:
            return None
        hd_stream = max(
            (s for s in metadata["streams"] if s["codec_type"] == "video"),
            key=lambda s: s["width"],
        )
        return VideoStream.from_metadata(hd_stream)

    def read_frame_opencv(self) -> Optional[cv.Mat]:
        cap = cv.VideoCapture(self.path.as_posix())
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
        stream = self.best_video_stream(self.get_metadata())
        if not stream:
            return None

        out, _ = (
            ffmpeg.input(str(self.path))
            .output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1, loglevel="quiet"
            )
            .global_args("-map", f"0:{stream.stream_index}")
            .run()
        )

        return np.frombuffer(out, np.uint8).reshape([stream.height, stream.width, 3])

    def read_frame(self):
        stream = self.best_video_stream(self.get_metadata())
        if not stream:
            return None

        process = (
            ffmpeg.input(str(self.path))
            .output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1, loglevel="quiet"
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


class Timeline:
    def __init__(self, video_path: Path):
        self.clips = self.get_clips(video_path)

    def get_clips(self, video_path: Path) -> List[Clip]:
        files = [Clip.build(file) for file in video_path.glob("*/*.mp4")]
        ans = sorted(files, key=lambda clip: clip.timestamp)
        return ans

    def __str__(self) -> str:
        return "\n".join(str(clip) for clip in self.clips)

    def search_clips(
        self, date_from: Optional[datetime] = None, date_to: Optional[datetime] = None
    ) -> List[Clip]:

        # Should improve this!!!! although the N is pretty small
        clips: List[Clip] = []
        for clip in self.clips:
            if date_from and clip.timestamp < date_from:
                continue
            if date_to and clip.timestamp > date_to:
                continue
            clips.append(clip)
        return clips

    def show(
        self, date_from: Optional[datetime] = None, date_to: Optional[datetime] = None
    ) -> None:

        filtered_clips = self.search_clips(date_from, date_to)
        k = 1
        for clip in filtered_clips[::k]:
            print(clip)

            frame = clip.read_frame()
            colors = dynamic_range(frame)
            print(colors)
            cv.imshow("frame", frame)

            if cv.waitKey(1) == ord("q"):
                break

        cv.destroyAllWindows()

    def create(
        self,
        output: Path,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> None:
        filtered_clips = self.search_clips(date_from, date_to)
        filtered_clips = filtered_clips[::10]
        print("num clips", len(filtered_clips))

        # Process in batches of 100
        batch_size = 100

        # Get dimensions from first clip
        first_frame = filtered_clips[0].read_frame()
        height, width = first_frame.shape[:2]

        # Setup ffmpeg process
        process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
                framerate=30,
            )
            .output(str(output), pix_fmt="yuv420p", vcodec="libx264", crf=23)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        try:
            for i in range(0, len(filtered_clips), batch_size):
                batch = filtered_clips[i : i + batch_size]
                print(
                    f"Processing batch {i//batch_size + 1}/{(len(filtered_clips)-1)//batch_size + 1}"
                )

                batch_frames = Parallel(n_jobs=-1, verbose=10)(
                    delayed(clip.read_frame)() for clip in batch
                )

                # Write frames to video
                for frame in batch_frames:
                    if frame is not None:
                        process.stdin.write(frame.tobytes())

        finally:
            # Cleanup
            process.stdin.close()
            process.wait()


def parse_datetime(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    date_str = date_str.strip().strip("-")
    fmt = "%Y%m%d-%H%M%S"  # YYYYMMDD-HHMMSS format

    if len(date_str) == 8:
        date_str = f"{date_str}-000000"
    if len(date_str) > 8 and len(date_str) < 15:
        date_str = date_str.ljust(15, "0")

    dt = datetime.strptime(date_str, fmt)
    return dt.astimezone()


def main():
    parser = argparse.ArgumentParser(description="Create a timelapse video")
    parser.add_argument(
        "source",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--date-from",
        help="Start date (inclusive) format YYYYMMDD-HHMMSS localtime",
        type=parse_datetime,
    )
    parser.add_argument(
        "--date-to",
        help="End date (inclusive) format YYYYMMDD-HHMMSS localtime",
        type=parse_datetime,
    )
    parser.add_argument("--create", help="Create timelapse video", action="store_true")

    args = parser.parse_args()

    # video_path = "/Volumes/Cameras/aqara_video/lumi1.54ef44457bc9"
    timeline = Timeline(Path(args.source))
    # print(timeline)

    if args.create:
        timeline.create(Path("output.mp4"), args.date_from, args.date_to)
    else:
        timeline.show(args.date_from, args.date_to)
        # print(timeline.clips[0].get_hd_track())


if __name__ == "__main__":
    main()
