from pathlib import Path

from aqara_video.core.factory import TimelineFactory

# Automatically detects provider
timeline = TimelineFactory.create_timeline(
    Path("/mnt/hdd/diegocaro/aqara_video/lumi1.54ef44457bc9")
)
print(f"camera_id:", timeline.camera_id)
print(f"number of clips:", len(timeline))


TimelineFactory.create_timeline(Path("/mnt/hdd/diegocaro/aqara_video"))
# # Or use a specific provider explicitly
# from aqara_video.providers.aqara import AqaraProvider
# timeline = Timeline(Path("/path/to/camera/directory"), AqaraProvider())
