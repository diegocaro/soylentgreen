from pathlib import Path

from video_footage.core.factory import TimelineFactory
from video_footage.core.video_reader import VideoReader

# # Automatically detects provider
timeline = TimelineFactory.create_timeline(
    Path("/mnt/hdd/diegocaro/aqara_video/lumi1.54ef44457bc9")
    # Path("/mnt/cameras/aqara_video/lumi1.54ef44457bc9")
)
# print(f"camera_id:", timeline.camera_id)
# print(f"number of clips:", len(timeline))

# for clip in timeline.clips[:10]:
# print(clip.timestamp, clip.path)


video = VideoReader(
    Path("/mnt/hdd/diegocaro/aqara_video/lumi1.54ef44457bc9/20250301/185800.mp4")
)
# print(video.probe)
frame = video.read_frame()
print(frame.shape)
# TimelineFactory.create_timeline(Path("/mnt/hdd/diegocaro/aqara_video"))
# # Or use a specific provider explicitly
# from aqara_video.providers.aqara import AqaraProvider
# timeline = Timeline(Path("/path/to/camera/directory"), AqaraProvider())
