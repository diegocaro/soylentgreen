import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI()
VIDEO_DIR = Path("/Users/diegocaro/Projects/soylentgreen/sample-videos/aqara_video")
CLIP_DURATION = timedelta(minutes=1)


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
      <div id="playerContainer" style="position:relative; width:720px; height:405px;">
        <video id="playerA" controls playsinline style="position:absolute; top:0; left:0; width:100%; height:100%; z-index:2; transition: opacity 300ms;" ></video>
        <video id="playerB" controls playsinline style="position:absolute; top:0; left:0; width:100%; height:100%; z-index:1; transition: opacity 300ms;" ></video>
      </div>

<script>
async function initTimeline() {
  const res = await fetch('/list');
  const clips = await res.json();
  const items = clips.map((c, i) => ({
    id: i,
    content: c.name,
    start: c.start,
    end: c.end
  }));

  const container = document.getElementById('timeline');
  const timeline = new vis.Timeline(container, items, {
    selectable: true,
    zoomMin: 1000 * 60 * 5,
    zoomMax: 1000 * 60 * 60 * 24,
    stack: false
  });

  // Click anywhere on the timeline
  // store clips globally so we can advance to next
  window._sg_clips = clips;
  window._sg_currentIndex = null;

  const playerA = document.getElementById('playerA');
  const playerB = document.getElementById('playerB');
  let active = 'A'; // which player is currently visible/active

  function activePlayer() { return active === 'A' ? playerA : playerB; }
  function inactivePlayer() { return active === 'A' ? playerB : playerA; }

  // helper to load into the inactive player, seek when metadata is ready, play and then swap
  async function loadAndPlay(path, offsetSeconds, setActiveIndex) {
    const target = inactivePlayer();
    target.pause();
    target.removeAttribute('src');
    target.src = '/video?path=' + encodeURIComponent(path);

    const onMeta = function() {
      // clamp offset
      const seek = Math.max(0, Math.min(offsetSeconds || 0, target.duration || 0));
      try { target.currentTime = seek; } catch (e) {}
      // start playing hidden player; once it starts, swap z-index so it's visible
      const onPlaying = function() {
        // bring target on top
        if (active === 'A') {
          playerB.style.zIndex = 3;
          playerA.style.zIndex = 1;
        } else {
          playerA.style.zIndex = 3;
          playerB.style.zIndex = 1;
        }
        // update active flag
        active = (target === playerA) ? 'A' : 'B';
        // pause the now inactive one
        const nowInactive = inactivePlayer();
        try { nowInactive.pause(); } catch (e) {}
        target.removeEventListener('playing', onPlaying);
      };
      target.addEventListener('playing', onPlaying);
      target.play().catch(() => {/* autoplay may be blocked if not user-initiated */});
      target.removeEventListener('loadedmetadata', onMeta);
    };
    target.addEventListener('loadedmetadata', onMeta);
  }

  container.onclick = async (event) => {
    const props = timeline.getEventProperties(event);
    if (!props.time) return;
    const t = props.time.toISOString();
    const resp = await fetch('/seek?time=' + encodeURIComponent(t));
    if (!resp.ok) return;
    const { path, offset } = await resp.json();

    // find index in our clips list
    const idx = clips.findIndex(c => c.path === path);
    if (idx !== -1) window._sg_currentIndex = idx;

    await loadAndPlay(path, offset || 0, idx);
  };

  // when the active clip finishes, try to play the next one using the double-buffer approach
  playerA.addEventListener('ended', async () => {
    if (active !== 'A') return; // only respond if A is active
    const idx = window._sg_currentIndex;
    if (idx === null) return;
    const next = idx + 1;
    if (next >= clips.length) return;
    window._sg_currentIndex = next;
    const nextClip = clips[next];
    await loadAndPlay(nextClip.path, 0, next);
  });

  playerB.addEventListener('ended', async () => {
    if (active !== 'B') return; // only respond if B is active
    const idx = window._sg_currentIndex;
    if (idx === null) return;
    const next = idx + 1;
    if (next >= clips.length) return;
    window._sg_currentIndex = next;
    const nextClip = clips[next];
    await loadAndPlay(nextClip.path, 0, next);
  });
}
initTimeline();
</script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/list")
def list_videos():
    clips = []
    for f in VIDEO_DIR.rglob("*.mp4"):
        start = extract_timestamp(f)
        if not start:
            continue
        end = start + CLIP_DURATION
        clips.append(
            {
                "name": f.name,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "path": str(f.relative_to(VIDEO_DIR)),
            }
        )
    clips.sort(key=lambda x: x["start"])
    return JSONResponse(clips)


@app.get("/video")
def get_video(path: str):
    full_path = VIDEO_DIR / path
    return FileResponse(full_path)


@app.get("/seek")
def seek(time: str = Query(...)):
    """
    Given an absolute time (ISO string), find which clip covers it and the offset in seconds.
    """
    target = datetime.fromisoformat(time)
    candidates = []
    for f in VIDEO_DIR.rglob("*.mp4"):
        start = extract_timestamp(f)
        if not start:
            continue
        end = start + CLIP_DURATION
        if start <= target < end:
            offset = (target - start).total_seconds()
            return JSONResponse(
                {"path": str(f.relative_to(VIDEO_DIR)), "offset": offset}
            )
    return JSONResponse({"error": "no clip found"}, status_code=404)
