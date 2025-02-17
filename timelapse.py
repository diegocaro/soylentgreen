import argparse
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import ffmpeg
import cv2


import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

PIXEL_FORMAT = "bgr24"
CRF_VALUE = 23  # Constant Rate Factor (0-51), lower is better, default: 23


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


def dynamic_range(frame: cv2.Mat) -> Dict[str, float]:
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
            "greeness": 0.0,
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

    b, g, r = cv2.split(frame)

    # Calculate greeness as the difference between green and other channels
    # Higher values indicate more grass-like colors
    greeness = np.mean(g) - (np.mean(r) + np.mean(b)) / 2
    # greeness = np.mean(frame[:, :, 1] - frame[:, :, 0])

    distribution = {
        "dynamic_range": dynamic_range * 100,  # Convert to percentage
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "greeness": greeness,
    }

    return distribution


def post_process(
    frame: cv2.Mat,
    features: Dict[str, float],
    green_threshold: float = 0.2,
) -> cv2.Mat:
    if frame is None:
        return None

    frame_copy = frame.copy()

    greeness = features.get("greeness", 0.0)
    if green_threshold > 0 and greeness < green_threshold:
        return None

    text = f"Greeness score: {greeness:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Calculate position (10 pixels from left, 20 from top)
    x = 10
    y = text_height + 10  # y is the bottom-left corner of the text

    # print(f"Text dimensions: {text_width}x{text_height} pixels (baseline: {baseline})")

    cv2.putText(
        frame_copy,
        text,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
    )
    return frame_copy


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

    def read_frame_opencv(self) -> Optional[cv2.Mat]:
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
        stream = self.best_video_stream(self.get_metadata())
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

    def read_frame(self) -> cv2.Mat:
        stream = self.best_video_stream(self.get_metadata())
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


class Timeline:
    def __init__(self, video_path: Path):
        self.video_path = video_path
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

    def create_timelapse(
        self,
        output: Path,
        clips: List[Clip],
        green_threshold: float = 0.2,
    ) -> None:
        # Process in batches of 100
        batch_size = 100

        # Get dimensions from first clip
        first_frame = clips[0].read_frame()
        height, width = first_frame.shape[:2]

        # Setup ffmpeg process
        encoder = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=PIXEL_FORMAT,
                s=f"{width}x{height}",
                framerate=30,
            )
            .output(
                str(output),
                pix_fmt="yuv420p",
                vcodec="libx264",
                crf=CRF_VALUE,
                loglevel="quiet",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        def process(clip: Clip) -> cv2.Mat:
            frame = clip.read_frame()
            features = dynamic_range(frame)
            return post_process(
                frame, green_threshold=green_threshold, features=features
            )

        try:
            for i in range(0, len(clips), batch_size):
                batch = clips[i : i + batch_size]
                print(
                    f"Processing batch {i//batch_size + 1}/{(len(clips)-1)//batch_size + 1}"
                )

                batch_frames = Parallel(n_jobs=-1, verbose=10, return_as="generator")(
                    delayed(process)(clip) for clip in batch
                )

                # Write frames to video
                for frame in batch_frames:
                    if frame is not None:
                        cv2.imshow("frame", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            exit()
                        encoder.stdin.write(frame.tobytes())

        finally:
            # Cleanup
            encoder.stdin.close()
            encoder.wait()


class VideoPlayer:
    def show(self, clips: List[Clip]) -> None:
        for clip in clips:
            print(clip)
            frame = clip.read_frame()
            colors = dynamic_range(frame)
            print(colors)
            self.draw(frame)

        cv2.destroyAllWindows()

    def draw(self, frame: cv2.Mat):
        if frame is None:
            return
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()

    def show_threaded(self, clips: List[Clip], num_workers: int = 4) -> None:
        frame_queue = Queue(maxsize=1000)

        def read_frame(clip: Clip):
            frame = clip.read_frame()
            frame_queue.put((clip, frame))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(read_frame, clip) for clip in clips]

            frames_processed = 0
            total_frames = len(filtered_clips)

            while frames_processed < total_frames:
                clip, frame = frame_queue.get()
                if frame is not None:
                    self.draw(frame)
                frames_processed += 1
                if frames_processed % 10 == 0:
                    print(clip)
                    print(f"Progress: {frames_processed}/{total_frames}")

    def show_threaded_old(self, clips: List[Clip]) -> None:
        def read_frames(clips: List[Clip], frame_queue: Queue):
            for clip in clips:
                frame = clip.read_frame()
                frame_queue.put(frame)

        frame_queue = Queue()
        workers = threading.Thread(
            target=read_frames,
            args=(clips, frame_queue),
            daemon=True,
        )
        workers.start()

        def draw_frames(frame_queue: Queue):
            while True:
                frame = frame_queue.get()
                self.draw(frame)

        draw_frames(frame_queue)

    def show_joblib(self, clips: List[Clip], green_threshold: float = 0.2) -> None:
        def process(clip: Clip) -> cv2.Mat:
            frame = clip.read_frame()
            features = dynamic_range(frame)
            return post_process(
                frame, green_threshold=green_threshold, features=features
            )

        res = Parallel(n_jobs=-1, verbose=10, return_as="generator")(
            delayed(process)(clip) for clip in clips
        )
        for frame in res:
            self.draw(frame)


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
    parser.add_argument(
        "--day", help="Day to process in format YYYYMMDD", type=parse_datetime
    )
    parser.add_argument("--output", help="Output file", type=Path)
    parser.add_argument("--skip", help="Sample clips", type=int, default=1)
    parser.add_argument(
        "--green-threshold", help="Greeness threshold", type=float, default=0.2
    )
    args = parser.parse_args()

    # video_path = "/Volumes/Cameras/aqara_video/lumi1.54ef44457bc9"
    timeline = Timeline(Path(args.source))
    # print(timeline)
    if args.day:
        date_from = args.day.replace(hour=0, minute=0, second=0)
        date_to = args.day.replace(hour=23, minute=59, second=59)
        args.date_from = date_from
        args.date_to = date_to
    clips = timeline.search_clips(args.date_from, args.date_to)
    print(f"Found {len(clips)} clips at {args.source}")
    filtered_clips = clips[:: args.skip]
    print(f"Using {len(filtered_clips)} clips")

    if args.output:
        timeline.create_timelapse(
            args.output, filtered_clips, green_threshold=args.green_threshold
        )
    else:
        player = VideoPlayer()
        player.show_joblib(filtered_clips, green_threshold=args.green_threshold)
        exit()


if __name__ == "__main__":
    main()
