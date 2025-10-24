import io
import os
import subprocess

import ffmpeg
from flask import Flask, Response, abort, send_file, send_from_directory

# List of video files (assumed to be in the current directory)
VIDEO_FILES = [
    "000000.mp4",
    "000100.mp4",
    "000200.mp4",
    "000301.mp4",
    "000401.mp4",
    "000501.mp4",
    "000601.mp4",
    "000700.mp4",
    "000800.mp4",
    "000900.mp4",
    "001000.mp4",
]


# Get the actual duration of each video file using ffmpeg.probe
def get_video_duration(filepath):
    try:
        probe = ffmpeg.probe(filepath)
        duration = float(probe["format"]["duration"])
        return duration
    except Exception as e:
        print(f"Error probing {filepath}: {e}")
        return 0


app = Flask(__name__)


# Serve the video.html file at the root URL
@app.route("/")
def index():
    return send_from_directory(".", "video.html")


@app.route("/playlist.m3u8")
def playlist():
    # Generate HLS playlist referencing each segment with actual durations
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        # We'll set TARGETDURATION to the ceiling of the max segment duration
        # We'll compute this below
        # "#EXT-X-TARGETDURATION:...",
        "#EXT-X-MEDIA-SEQUENCE:0",
    ]

    durations = []
    for file in VIDEO_FILES:
        filepath = os.path.join("data", file)
        duration = get_video_duration(filepath)
        durations.append(duration)

    target_duration = int(max(durations)) if durations else 1
    lines.insert(3, f"#EXT-X-TARGETDURATION:{target_duration}")
    for file, duration in zip(VIDEO_FILES, durations):
        lines.append(f"#EXTINF:{duration:.3f},")
        lines.append(f"/segment/{file}")
    lines.append("#EXT-X-ENDLIST")
    return Response("\n".join(lines), mimetype="application/vnd.apple.mpegurl")


# Serve fMP4 segments directly
@app.route("/segment/<path:filename>")
def segment_mp4(filename):
    if filename not in VIDEO_FILES:
        abort(404)
    filepath = os.path.join("data", filename)
    start_time = 0

    # Use ffmpeg to extract the segment and stream as fMP4
    # Get the actual duration for this file
    actual_duration = get_video_duration(filepath)
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        filepath,
        "-t",
        str(actual_duration),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "pipe:1",
    ]

    def generate():
        print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while True:
                chunk = p.stdout.read(4096)
                print(".", end="", flush=True)
                if not chunk:
                    break
                yield chunk
        finally:
            p.stdout.close()
            p.terminate()
            print("Segment stream ended.")

    return Response(generate(), mimetype="video/mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
