import logging
import time
from pathlib import Path

import cv2
import numpy as np

from aqara_video.core.video_reader import VideoReader

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class YellowBoxDetector:
    # Yellow range (HSV)
    LOWER_YELLOW = np.array([25, 250, 250])
    UPPER_YELLOW = np.array([35, 255, 255])
    MIN_YELLOW_PIXELS = 10

    FRAME_SKIP = 5  # analyze every 5th frame
    ACTIVE_INTERVAL = 2.0  # seconds

    def __init__(
        self,
        lower_yellow: np.ndarray = LOWER_YELLOW,
        upper_yellow: np.ndarray = UPPER_YELLOW,
        min_yellow_pixels: int = MIN_YELLOW_PIXELS,
        frame_skip: int = FRAME_SKIP,
        active_interval: float = ACTIVE_INTERVAL,
    ):
        self._lower_yellow = lower_yellow
        self._upper_yellow = upper_yellow
        self._min_yellow_pixels = min_yellow_pixels
        self._frame_skip = frame_skip
        self._active_interval = active_interval

    def detect(self, frame_bgr: np.ndarray) -> bool:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower_yellow, self._upper_yellow)
        # Uncomment if you want to find the bounding box around all yellow pixels
        # coords = cv2.findNonZero(mask)
        # if coords is not None:
        #     x, y, w, h = cv2.boundingRect(coords)
        #     print(
        #         f"Frame {videoframe.frame_id} - t={videoframe.t:.2f}s: Bounding box: x={x}, y={y}, w={w}, h={h}"
        #     )
        return cv2.countNonZero(mask) > self._min_yellow_pixels

    def timestamps(self, video_path: Path) -> list[float]:
        video = VideoReader(video_path)
        logger.debug(f"Processing video: {video_path}, duration: {video.duration:.2f}s")
        timestamps: list[float] = []
        for videoframe in video.frames(frame_skip=self._frame_skip):
            if videoframe.frame_id % 100 == 0:
                logger.debug(
                    f"Analyzing frame {videoframe.frame_id} at t={videoframe.t:.2f}s"
                )
            if self.detect(videoframe.frame):
                timestamps.append(videoframe.t)
        return timestamps

    def intervals_from_timestamps(
        self, timestamps: list[float]
    ) -> list[tuple[float, float]]:
        if not timestamps:
            return []
        merged: list[tuple[float, float]] = []
        start = last = timestamps[0]
        for t in timestamps[1:]:
            if t - last <= self._active_interval:
                last = t
            else:
                merged.append((start, last + self._active_interval))
                start = last = t
        merged.append((start, last + self._active_interval))
        return merged

    def predict(self, video_path: Path) -> list[tuple[float, float]]:
        timestamps = self.timestamps(video_path)
        logger.debug(f"Detected {len(timestamps)} frames with yellow boxes")
        intervals = self.intervals_from_timestamps(timestamps)
        logger.debug(f"Merged into {len(intervals)} intervals")
        return intervals


def test():
    logging.basicConfig(level=logging.DEBUG)
    VIDEO_PATH = "/Volumes/Cameras/aqara_video/lumi1.54ef44603857/20251023/102559.mp4"
    detector = YellowBoxDetector()
    start_time = time.time()
    intervals = detector.predict(Path(VIDEO_PATH))
    end_time = time.time()
    logger.debug(f"Processing time: {end_time - start_time:.2f} seconds")

    logger.debug("Detected yellow box intervals (seconds):")
    for s, e in intervals:
        print(f"{s:.2f} → {e:.2f}")


if __name__ == "__main__":
    test()
