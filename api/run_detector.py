import argparse
import logging

from api.config import BOX_DETECTION_FILE, SCAN_RESULT_FILE, VIDEO_DIR
from api.detector import YellowBoxDetector
from api.models import LabeledInterval, ScanResult, VideoDetectionSummary

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run yellow box detection on videos.")
    parser.add_argument(
        "--camera",
        "-c",
        nargs="*",
        help="Camera ID(s) to process. If not set, all cameras will be processed.",
    )
    args = parser.parse_args()

    detections = load_detections()
    yellow_box_detector = YellowBoxDetector()
    scan_result = ScanResult.model_validate_json(SCAN_RESULT_FILE.read_text())

    selected_cameras = set(args.camera) if args.camera else None

    for camera_id, video_list in scan_result.cameras.items():
        if selected_cameras and camera_id not in selected_cameras:
            continue
        logger.info(f"Processing camera: {camera_id}")
        for segment in video_list.segments:
            video_path = VIDEO_DIR / segment.path

            # Check if already processed
            if segment.path in detections.detections:
                logger.info(f"Skipping {segment.path} (already processed)")
                continue

            logger.info(f"Analyzing video: {video_path}")
            try:
                intervals = yellow_box_detector.predict(video_path)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                intervals = []

            # Store detection result
            box_intervals = [
                LabeledInterval(label="yellow_box", start=start, end=end)
                for start, end in intervals
            ]
            detections.detections[segment.path] = box_intervals

            if intervals:
                logger.info(
                    f"Detected {len(intervals)} yellow box intervals in {segment.path}"
                )
            else:
                logger.info(f"No yellow boxes detected in {segment.path}")

            # Save after each video to avoid losing work
            save_detections(detections)

    logger.info(
        f"Processing complete. Total videos processed: {len(detections.detections)}"
    )
