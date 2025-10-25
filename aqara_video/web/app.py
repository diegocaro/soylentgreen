from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from aqara_video.web.service import Service

VIDEO_DIR = Path("/Users/diegocaro/Projects/soylentgreen/sample-videos/aqara_video")

app = FastAPI()


def get_service() -> Service:
    return Service(base_path=VIDEO_DIR)


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/list")
def list_videos(service: Service = Depends(get_service)):
    clips = service.list_videos()
    return JSONResponse(clips)


@app.get("/video")
def get_video(path: str, service: Service = Depends(get_service)):
    full_path = service.get_video_path(path)
    return FileResponse(full_path)


@app.get("/seek")
def seek(time: str = Query(...), service: Service = Depends(get_service)):
    target = datetime.fromisoformat(time)
    result = service.seek(target)
    if result:
        return JSONResponse(result)
    return JSONResponse({"error": "no clip found"}, status_code=404)
