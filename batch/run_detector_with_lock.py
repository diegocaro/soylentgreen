import argparse
import logging
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

from api.config import VIDEO_DIR
from api.schemas import LabeledInterval, ScanResult
from batch.lock_claim import (
    FileClaimLock,
    FileClaimLockError,
    atomic_write_json,
    now_iso_safe,
    try_claim,
)
from batch.scan import ScanManager
from detection.yellow_box_detector import YellowBoxDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionTask:
    def __init__(
        self,
        model: YellowBoxDetector,
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

    def claim_video(self, full_video_path: Path):
        return try_claim(full_video_path, self.worker_id)

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

    def run(self, video_path: Path, force: bool = False):
        full_video_path = self.get_full_video_path(video_path)
        result_dir = self.get_result_dir(full_video_path)
        if self.is_already_processed(result_dir, force):
            logger.info(
                f"Skipping {video_path} (already processed by model {self.model_name} {self.model_version})"
            )
            return video_path, None

        try:
            with FileClaimLock(full_video_path, self.worker_id) as claim:
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
        except FileClaimLockError:
            logger.info(f"Skipping {video_path} (already claimed by another worker)")
            return video_path, None
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return video_path, None
        return video_path, intervals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run yellow box detection on videos.")
    parser.add_argument(
        "--camera",
        "-c",
        nargs="*",
        help="Camera ID(s) to process. If not set, all cameras will be processed.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs to run (default: 1)",
    )
    parser.add_argument(
        "--worker-id",
        "-w",
        type=str,
        default="worker-1",
        help="Worker ID for claiming locks (default: worker-1)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reprocessing even if results exist.",
    )
    parser.add_argument(
        "--only-locked",
        action="store_true",
        help="Process only files with existing lock files (possibly stuck from another run)",
    )
    args = parser.parse_args()

    yellow_box_detector = YellowBoxDetector()

    manager = ScanManager(VIDEO_DIR)
    cameras = manager.scan_cameras()
    logger.info("Cameras found:")
    for camera in cameras:
        logger.info(f" - {camera}")

    selected_cameras = set(args.camera) if args.camera else cameras

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

    video_task = PredictionTask(
        yellow_box_detector, VIDEO_DIR, args.worker_id, "labels", "results"
    )

    Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(video_task.run)(
            segment.path,
            args.force,
        )
        for _camera_id, segment in segments_to_process
    )

    logger.info(
        f"Processing complete. Total videos processed: {len(segments_to_process)}"
    )
