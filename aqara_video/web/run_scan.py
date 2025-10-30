import logging
from datetime import datetime

from aqara_video.web.config import SCAN_RESULT_FILE, VIDEO_DIR
from aqara_video.web.models import ScanResult
from aqara_video.web.scan import ScanManager

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    manager = ScanManager(VIDEO_DIR)
    cameras = manager.scan_cameras()
    logger.info("Cameras found:")
    for camera in cameras:
        logger.info(f" - {camera}")

    scan = {camera: manager.scan_videos(camera) for camera in cameras}
    result = ScanResult(cameras=scan, scanned_at=datetime.now())

    with open(SCAN_RESULT_FILE, "w") as f:
        f.write(result.model_dump_json(indent=2))
    logger.info(f"Scan result saved to {SCAN_RESULT_FILE}")
