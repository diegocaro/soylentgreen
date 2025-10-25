import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI()
VIDEO_DIR = Path("/Users/diegocaro/Projects/soylentgreen/sample-videos/aqara_video")


def extract_timestamp(path: Path) -> datetime:
    """Extract timestamp from Aqara path format."""
    # lumi1.54ef44457bc9/20250207/082900.mp4 -> 2025-02-07 08:29:00 UTC
    hhmmss = path.stem
    yyyymmdd = path.parent.stem
    timestamp = datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
    timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
    return timestamp_utc.astimezone()  # Convert to local timezone


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <html>
    <head>
      <title>NAS Timeline Viewer</title>
      <script src="https://unpkg.com/vis-timeline@7.7.0/standalone/umd/vis-timeline-graph2d.min.js"></script>
      <link href="https://unpkg.com/vis-timeline@7.7.0/styles/vis-timeline-graph2d.min.css" rel="stylesheet" />
      <style>
        body { font-family: sans-serif; padding: 20px; }
        #timeline { border: 1px solid #ccc; height: 200px; }
        video { margin-top: 20px; width: 720px; height: 405px; }
      </style>
    </head>
    <body>
      <h2>NAS Video Timeline</h2>
      <div id="timeline"></div>
      <video id="player" controls></video>

      <script>
        async function loadTimeline() {
          const res = await fetch('/list');
          const files = await res.json();
          const items = files.map((f, i) => ({
            id: i,
            content: f.name,
            start: f.timestamp,
            path: f.path
          }));

          const container = document.getElementById('timeline');
          const timeline = new vis.Timeline(container, items, {
            zoomMin: 1000 * 60 * 5, // 5 min
            zoomMax: 1000 * 60 * 60 * 24,
            stack: false
          });

          timeline.on('select', function (props) {
            if (props.items.length > 0) {
              const item = items.find(i => i.id === props.items[0]);
              document.getElementById('player').src = '/video?path=' + encodeURIComponent(item.path);
            }
          });
        }
        loadTimeline();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/list")
def list_videos():
    data = []
    for f in VIDEO_DIR.rglob("*.mp4"):
        ts = extract_timestamp(f)
        if ts:
            data.append(
                {
                    "name": f.name,
                    "timestamp": ts.isoformat(),
                    "path": str(f.relative_to(VIDEO_DIR)),
                }
            )
    data.sort(key=lambda x: x["timestamp"])
    return JSONResponse(data)


@app.get("/video")
def get_video(path: str):
    full_path = VIDEO_DIR / path
    return FileResponse(full_path)
