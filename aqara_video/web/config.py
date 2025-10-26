import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCAN_RESULT_FILE = Path("scan_result.json")

# VIDEO_DIR = Path("/Users/diegocaro/Projects/soylentgreen/sample-videos/aqara_video")
VIDEO_DIR = Path(os.environ["VIDEO_DIR"])
CLIP_DURATION = timedelta(minutes=1)

camera_map_str = os.getenv("CAMERA_MAP", "")
CAMERA_MAP = dict(item.split(":") for item in camera_map_str.split(",") if item)
