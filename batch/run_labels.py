import logging
from datetime import datetime

from api.config import BOX_DETECTION_FILE, LABELS_TIMELINE_FILE, SCAN_RESULT_FILE
from api.schemas import (
    CameraLabels,
    LabelsByCamera,
    LabelTimeline,
    ScanResult,
    TimeInterval,
    VideoDetectionSummary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def absolute_intervals(
    detections: VideoDetectionSummary, scan_result: ScanResult
) -> dict[str, CameraLabels]:
    """
    Convert video-relative detections to absolute timestamps, grouped by camera ID and label.
    """
    absolute_by_camera = {}

    for camera_id, video_list in scan_result.cameras.items():
        for segment in video_list.segments:
            interval_list = detections.detections.get(segment.path, [])
            if not interval_list:
                continue

            segment_start = segment.start

            for interval in interval_list:
                start_dt = segment_start.replace(
                    microsecond=0
                ) + datetime.resolution * int(interval.start * 1_000_000)
                end_dt = segment_start.replace(
                    microsecond=0
                ) + datetime.resolution * int(interval.end * 1_000_000)

                if camera_id not in absolute_by_camera:
                    absolute_by_camera[camera_id] = {}

                if interval.label not in absolute_by_camera[camera_id]:
                    absolute_by_camera[camera_id][interval.label] = []

                absolute_by_camera[camera_id][interval.label].append(
                    TimeInterval(
                        start=start_dt,
                        end=end_dt,
                    )
                )

    # Convert to proper model structure
    cameras = {}
    for camera_id, labels_dict in absolute_by_camera.items():
        label_timelines = {
            label: LabelTimeline(intervals=intervals)
            for label, intervals in labels_dict.items()
        }
        cameras[camera_id] = CameraLabels(labels=label_timelines)

    return cameras


if __name__ == "__main__":
    box_detection = VideoDetectionSummary.model_validate_json(
        BOX_DETECTION_FILE.read_text()
    )
    scan_result = ScanResult.model_validate_json(SCAN_RESULT_FILE.read_text())

    cameras = absolute_intervals(box_detection, scan_result)

    logger.info(f"Found intervals for {len(cameras)} cameras")

    labels_by_camera = LabelsByCamera(cameras=cameras)

    LABELS_TIMELINE_FILE.write_text(labels_by_camera.model_dump_json(indent=2))

    logger.info(f"Labels by camera saved to {LABELS_TIMELINE_FILE}")
