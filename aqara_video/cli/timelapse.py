import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

import cv2
import numpy as np
from joblib import Parallel, delayed

from aqara_video.core.clip import Clip
from aqara_video.core.factory import TimelineFactory
from aqara_video.core.types import ImageCV
from aqara_video.core.video_writer import VideoWriter


def is_graphical_environment():
    """Detect if running in a graphical environment"""
    if not os.environ.get("DISPLAY"):
        return False
    return True


def dynamic_range(frame: ImageCV) -> Dict[str, float]:
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
    frame: ImageCV,
    features: Dict[str, float],
    green_threshold: float = 0.2,
) -> ImageCV:
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


class Timelapse:
    def __init__(self):
        self.has_display = is_graphical_environment()
        if self.has_display:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 640, 480)

    def create_timelapse(
        self,
        output: Path,
        clips: List[Clip],
        green_threshold: float = 0.2,
    ) -> None:
        # Process in batches of 100
        batch_size = 100

        # Get dimensions from first clip
        clip = clips[0]

        # Setup the video writer
        writer = VideoWriter(output, clip.width, clip.height)

        def process(clip: Clip) -> ImageCV:
            frame = clip.read_frame()
            features = dynamic_range(frame)
            return post_process(
                frame, green_threshold=green_threshold, features=features
            )

        try:
            writer.open()
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
                        writer.write_frame(frame)
                        if self.has_display:
                            cv2.imshow("frame", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                exit()

        finally:
            # Cleanup
            writer.close()


class VideoPlayer:
    def show(self, clips: List[Clip]) -> None:
        for clip in clips:
            print(clip)
            frame = clip.read_frame()
            colors = dynamic_range(frame)
            print(colors)
            self.draw(frame)

        cv2.destroyAllWindows()

    def draw(self, frame: ImageCV):
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
        def process(clip: Clip) -> ImageCV:
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

    if len(date_str) == 10:
        # YYYY-MM-DD format
        date_str = date_str.replace("-", "")
    if len(date_str) == 8:
        # YYYYMMDD format
        date_str = f"{date_str}-000000"
    if len(date_str) > 8 and len(date_str) < 15:
        # Add missing seconds
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
        "--day", help="Day to process in format YYYY-MM-DD", type=parse_datetime
    )
    parser.add_argument("--output", help="Output file", type=Path)
    parser.add_argument("--skip", help="Sample clips", type=int, default=1)
    parser.add_argument(
        "--green-threshold", help="Greeness threshold", type=float, default=0.2
    )
    args = parser.parse_args()

    timeline = TimelineFactory.create_timeline(Path(args.source))
    if args.day:
        date_from = args.day.replace(hour=0, minute=0, second=0)
        date_to = args.day.replace(hour=23, minute=59, second=59)
        args.date_from = date_from
        args.date_to = date_to
    clips = timeline.search_clips(args.date_from, args.date_to)
    print(f"Found {len(clips)} clips at {args.source}")
    filtered_clips = clips[:: args.skip]
    print(f"Using {len(filtered_clips)} clips")

    timelapse = Timelapse()
    if args.output:
        timelapse.create_timelapse(
            args.output, filtered_clips, green_threshold=args.green_threshold
        )
    else:
        player = VideoPlayer()
        player.show_joblib(filtered_clips, green_threshold=args.green_threshold)
        exit()


if __name__ == "__main__":
    main()
