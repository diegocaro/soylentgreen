import logging
from datetime import datetime

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, JSONResponse

from aqara_video.web.config import SCAN_RESULT_FILE, VIDEO_DIR
from aqara_video.web.models import ScanResult
from aqara_video.web.service import Service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()


def get_service() -> Service:
    scan_result = ScanResult.model_validate_json(SCAN_RESULT_FILE.read_text())

    return Service(root_dir=VIDEO_DIR, scan_result=scan_result)


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/cameras")
def list_cameras(service: Service = Depends(get_service)):
    return service.list_cameras()


@app.get("/list")
def list_videos(camera_id: str, service: Service = Depends(get_service)):
    return service.list_videos(camera_id=camera_id)


@app.get("/video")
def get_video(path: str, service: Service = Depends(get_service)):
    full_path = service.get_video_path(path)
    return FileResponse(full_path)


@app.get("/seek")
def seek(camera_id: str, time: str, service: Service = Depends(get_service)):
    target = datetime.fromisoformat(time)
    result = service.seek(camera_id, target)
    if result:
        return result
    return JSONResponse({"error": "no clip found"}, status_code=404)
