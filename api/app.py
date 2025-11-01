import logging
from datetime import datetime

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, JSONResponse

from api.config import CAMERA_MAP, LABELS_TIMELINE_FILE, SCAN_RESULT_FILE, VIDEO_DIR
from api.models import (
    CameraInfo,
    CameraLabels,
    LabelsByCamera,
    ScanResult,
    TimeInterval,
)
from api.service import Service

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:      %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI()


def get_service() -> Service:
    scan_result = ScanResult.model_validate_json(SCAN_RESULT_FILE.read_text())
    labels_timeline = LabelsByCamera.model_validate_json(
        LABELS_TIMELINE_FILE.read_text()
    )
    return Service(
        root_dir=VIDEO_DIR,
        scan_result=scan_result,
        labels_timeline=labels_timeline,
        camera_map=CAMERA_MAP,
    )


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/cameras")
def list_cameras(service: Service = Depends(get_service)) -> list[CameraInfo]:
    return service.list_cameras()


@app.get("/list")
def list_videos(camera_id: str, service: Service = Depends(get_service)):
    return service.list_videos(camera_id=camera_id)


@app.get("/video")
def get_video(path: str, service: Service = Depends(get_service)):
    full_path = service.get_video_path(path)
    return FileResponse(full_path)


@app.get("/seek")
def seek(
    camera_id: str,
    time: str,
    return_next: bool = False,
    service: Service = Depends(get_service),
):
    target = datetime.fromisoformat(time)
    result = service.seek(camera_id, target, return_next=return_next)
    if result:
        return result
    return JSONResponse({"error": "no clip found"}, status_code=404)


@app.get("/labels")
def get_labels(camera_id: str, service: Service = Depends(get_service)) -> CameraLabels:
    return service.get_labels_timeline(camera_id)


@app.get("/list-intervals")
def list_intervals(
    camera_id: str, service: Service = Depends(get_service)
) -> list[TimeInterval]:
    return service.list_intervals(camera_id=camera_id)
