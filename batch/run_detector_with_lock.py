import argparse
import logging
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

from api.config import BOX_DETECTION_FILE, SCAN_RESULT_FILE, VIDEO_DIR
from api.schemas import LabeledInterval, ScanResult, VideoDetectionSummary
from batch.lock_claim import (
    FileClaimLock,
    FileClaimLockError,
    atomic_write_json,
    now_iso_safe,
)
from batch.scan import ScanManager
from detection.model_protocol import ModelProtocol
from detection.yellow_box_detector import YellowBoxDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_detections() -> VideoDetectionSummary:
    """Load existing detections from JSON file."""
    if BOX_DETECTION_FILE.exists():
        logger.info(f"Loading existing detections from {BOX_DETECTION_FILE}")
        return VideoDetectionSummary.model_validate_json(BOX_DETECTION_FILE.read_text())
    return VideoDetectionSummary(detections={})


def save_detections(detections: VideoDetectionSummary):
    """Save detections to JSON file."""
    logger.info(f"Saving detections to {BOX_DETECTION_FILE}")
    BOX_DETECTION_FILE.write_text(detections.model_dump_json(indent=2))


class PredictionTask:
    def __init__(
        self,
        model: ModelProtocol,
        video_dir: Path,
        worker_id: str = "worker-1",
        prefix_file: str = "labels",
        suffix_folder: str = "results",
    ):
        self.yellow_box_detector = model
        self.video_dir = video_dir
        self.worker_id = worker_id
        self.prefix_file = prefix_file
        self.suffix_folder = suffix_folder
        self.model_name = model.model_name
        self.model_version = model.model_version

    def get_full_video_path(self, video_path: Path):
        return self.video_dir / video_path

    def get_result_dir(self, full_video_path: Path):
        return full_video_path.with_name(f"{full_video_path.name}.{self.suffix_folder}")

    def is_already_processed(self, result_dir: Path, force: bool):
        result_pattern = (
            f"{self.prefix_file}-{self.model_name}-{self.model_version}-*.json"
        )
        return (
            result_dir.exists() and any(result_dir.glob(result_pattern)) and not force
        )

    def save_result(self, video_path: Path, box_intervals, result_dir: Path):
        timestamp = now_iso_safe()
        result_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "worker_id": self.worker_id,
            "video_path": str(video_path),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "labels": [
                {"label": interval.label, "start": interval.start, "end": interval.end}
                for interval in box_intervals
            ],
        }
        result_path = (
            result_dir
            / f"{self.prefix_file}-{self.model_name}-{self.model_version}-{timestamp}.json"
        )
        atomic_write_json(result_path, result)
        logger.info(f"Stored prediction in {result_path}")

    def run(self, video_path: Path, force: bool = False, post_detect_callback=None):
        full_video_path = self.get_full_video_path(video_path)
        result_dir = self.get_result_dir(full_video_path)
        if self.is_already_processed(result_dir, force):
            logger.info(
                f"Skipping {video_path} (already processed by model {self.model_name} {self.model_version})"
            )
            return video_path, None

        try:
            with FileClaimLock(full_video_path, self.worker_id):
                logger.info(f"Claimed {video_path} for processing by {self.worker_id}")
            logger.info(f"Analyzing video: {full_video_path}")
            intervals = self.yellow_box_detector.predict(full_video_path)

            if isinstance(intervals, list):
                box_intervals = [
                    LabeledInterval(label="yellow_box", start=start, end=end)
                    for start, end in intervals
                ]
            else:
                raise ValueError("Unexpected intervals format from predictor")

            if intervals:
                logger.info(
                    f"Detected {len(intervals)} yellow box intervals in {video_path}"
                )
            else:
                logger.info(f"No yellow boxes detected in {video_path}")

            self.save_result(video_path, box_intervals, result_dir)
        except FileClaimLockError as e:
            logger.info(f"Skipping {video_path}: {e}")
            return video_path, None
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return video_path, None

        if post_detect_callback:
            post_detect_callback(video_path, box_intervals)
        return video_path, box_intervals


def create_old_scan_callback(detections: VideoDetectionSummary):
    def callback(video_path: Path, box_intervals):
        detections.detections[str(video_path)] = box_intervals
        save_detections(detections)

    return callback


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Yellow Box Detector on videos with file locking."
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default="worker-1",
        help="Identifier for the worker process.",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=str,
        nargs="*",
        help="List of camera IDs to process. If not provided, all cameras are processed.",
    )
    parser.add_argument(
        "--only-locked",
        action="store_true",
        help="Process only videos that are already locked.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of videos even if results already exist.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--old-scan",
        action="store_true",
        help="Use existing scan result instead of rescanning videos.",
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()

    model = YellowBoxDetector()

    manager = ScanManager(VIDEO_DIR)
    cameras = manager.scan_cameras()
    logger.info("Cameras found:")
    for camera in cameras:
        logger.info(f" - {camera}")

    selected_cameras = set(args.camera) if args.camera else cameras

    old_scan_callback = None
    if args.old_scan:
        logger.info(f"Loading existing scan result at {SCAN_RESULT_FILE}")
        scan_result = ScanResult.model_validate_json(SCAN_RESULT_FILE.read_text())
        detections = load_detections()
        old_scan_callback = create_old_scan_callback(detections)
    else:
        scan = {camera: manager.scan_videos(camera) for camera in selected_cameras}
        scan_result = ScanResult(cameras=scan, scanned_at=datetime.now())

    segments_to_process = []
    for camera_id, video_list in scan_result.cameras.items():
        if selected_cameras and camera_id not in selected_cameras:
            continue
        for segment in video_list.segments:
            video_path = VIDEO_DIR / segment.path
            lock_path = video_path.with_name(video_path.name + ".lock")
            if args.only_locked:
                if not lock_path.exists():
                    continue
            segments_to_process.append((camera_id, segment))

    video_task = PredictionTask(model, VIDEO_DIR, args.worker_id, "labels", "results")

    Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(video_task.run)(
            segment.path,
            args.force,
            post_detect_callback=old_scan_callback,
        )
        for _camera_id, segment in segments_to_process
    )

    logger.info(
        f"Processing complete. Total videos processed: {len(segments_to_process)}"
    )
