import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOTDIR = Path(__file__).parent.parent.resolve()
DATADIR = ROOTDIR / "data"

SCAN_RESULT_FILE = DATADIR / "scan_result.json"
BOX_DETECTION_FILE = DATADIR / "box_detection.json"
LABELS_TIMELINE_FILE = DATADIR / "labels_timeline.json"

# VIDEO_DIR = Path("/Users/diegocaro/Projects/soylentgreen/sample-videos/aqara_video")
VIDEO_DIR = Path(os.environ["VIDEO_DIR"]).resolve()
CLIP_DURATION = timedelta(minutes=1)

camera_map_str = os.getenv("CAMERA_MAP", "")
CAMERA_MAP = dict(item.split(":") for item in camera_map_str.split(",") if item)
DEFAULT_CAMERA_ID = os.getenv("DEFAULT_CAMERA_ID", "")
