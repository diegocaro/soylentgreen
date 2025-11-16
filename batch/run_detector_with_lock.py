import argparse
import logging
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

from api.config import VIDEO_DIR
from api.schemas import LabeledInterval, ScanResult
from batch.lock_claim import atomic_write_json, now_iso_safe, release_claim, try_claim
from batch.scan import ScanManager
from detection.yellow_box_detector import YellowBoxDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_segment(
    yellow_box_detector: YellowBoxDetector,
    video_dir: Path,
    video_path: Path,
    worker_id: str = "worker-1",
    prefix_file: str = "labels",
    suffix_folder: str = "results",
    force: bool = False,
):
    full_video_path = video_dir / video_path
    # Check if already processed
    result_dir: Path = full_video_path.with_name(
        f"{full_video_path.name}.{suffix_folder}"
    )
    model_name = yellow_box_detector.model_name
    model_version = yellow_box_detector.model_version
    # Use glob to check for any result file for this model/version
    result_pattern = f"{prefix_file}-{model_name}-{model_version}-*.json"
    if result_dir.exists() and any(result_dir.glob(result_pattern)) and not force:
        logger.info(
            f"Skipping {video_path} (already processed by model {model_name} {model_version})"
        )
        return video_path, None

    # Try to claim the video for exclusive processing
    claim = try_claim(full_video_path, worker_id)
    if not claim:
        logger.info(f"Skipping {video_path} (already claimed by another worker)")
        return video_path, None

    lock_path, _info = claim
    logger.info(f"Claimed {video_path} for processing by {worker_id}")
    try:
        logger.info(f"Analyzing video: {full_video_path}")
        intervals = yellow_box_detector.predict(full_video_path)

        if isinstance(intervals, tuple) and len(intervals) == 2:
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

        # Store predictions in a sibling results folder
        timestamp = now_iso_safe()
        result_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "worker_id": worker_id,
            "video_path": str(video_path),
            "model_name": model_name,
            "model_version": model_version,
            "labels": [
                {"label": interval.label, "start": interval.start, "end": interval.end}
                for interval in box_intervals
            ],
        }
        result_path: Path = (
            result_dir
            / f"{prefix_file}-{result['model_name']}-{result['model_version']}-{timestamp}.json"
        )
        atomic_write_json(result_path, result)
        logger.info(f"Stored prediction in {result_path}")
    finally:
        release_claim(lock_path)


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

    Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(process_segment)(
            yellow_box_detector,
            VIDEO_DIR,
            segment.path,
            args.worker_id,
            "labels",
            "results",
            args.force,
        )
        for camera_id, segment in segments_to_process
    )

    logger.info(
        f"Processing complete. Total videos processed: {len(segments_to_process)}"
    )
