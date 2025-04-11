import threading
from pathlib import Path
from queue import Queue
from typing import Dict, Optional

import cv2
from IPython.display import display
from ipywidgets import Dropdown, HBox, Image, Layout, SelectionSlider, Text, VBox

from aqara_video.core.clip import Clip
from aqara_video.core.factory import TimelineFactory

MAPPING_CAMERAS = {
    "lumi1.54ef44457bc9": "Pasto",
    "lumi1.54ef44603857": "Living",
}


class JupyterViewer:
    def __init__(
        self, videos_path: Path, cameras_names: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Aqara Viewer

        Args:
            videos_path (str): Path to the directory containing video files
            cameras_names (dict, optional): Mapping of camera names to human-readable names
        """
        self.videos_path = videos_path

        if cameras_names is None:
            cameras_names = MAPPING_CAMERAS
        self.mapping_cameras = cameras_names

        # Create widgets
        self.img_widget = Image()

        self.dropdown_cameras = Dropdown(
            options=self.cameras,
            description="Camera:",
        )

        self.text_box = Text(description="File:", layout=Layout(width="1000px"))
        self.dropdown_days = Dropdown(description="Day:")
        self.selector_minutes = SelectionSlider(
            options=[("No videos available", "")],
            description="Minute:",
            disabled=True,
            layout=Layout(width="1000px"),
        )

        self.clips = []
        self.clip_queue: Queue[Clip] = Queue(maxsize=1000)

        # Set up event handlers
        self.dropdown_cameras.observe(self.update_days, names="value")
        self.dropdown_days.observe(self.update_minutes, names="value")
        self.selector_minutes.observe(self.update_video, names="value")

        # Start video processing thread
        self.loop = threading.Thread(
            target=self.refresh_loop,
            args=(self.clip_queue,),
            daemon=True,
        )
        self.loop.start()

        # Initial update
        self.update_days()

    @property
    def cameras(self):
        return [
            (self.mapping_cameras.get(c.name, c.name), c)
            for c in Path(self.videos_path).glob("lumi1.*")
        ]

    def refresh_loop(self, clip_q: Queue[Clip]):
        """Thread function to process video clips"""
        while True:
            selected_clip = clip_q.get()
            while not clip_q.empty():
                try:
                    selected_clip = clip_q.get_nowait()
                except:
                    break

            self.text_box.value = f"{selected_clip.path} fps={selected_clip.fps:.2f}, {selected_clip.width}x{selected_clip.height}"

            for frame_id, frame in selected_clip.frames():
                if not clip_q.empty():
                    break  # New clip available, stop current playback

                if frame_id % 60 == 0:
                    _, jpeg = cv2.imencode(".jpg", frame)
                    jpeg_bytes = jpeg.tobytes()
                    self.img_widget.value = jpeg_bytes
                    break

            # This will force to continue to the next clip, very HACKY
            if self.selector_minutes.index + 1 < len(self.selector_minutes.options):
                self.selector_minutes.index += 1

    def update_days(self, *args):
        """Update the days dropdown based on camera selection"""
        self.clips = TimelineFactory.create_timeline(self.dropdown_cameras.value).clips

        days = sorted(list(set(c.timestamp.date() for c in self.clips)))
        self.dropdown_days.options = days
        # Also update minutes when days change
        self.update_minutes()

    def update_minutes(self, *args):
        """Update the minutes selection slider based on day selection"""
        if self.dropdown_days.value is None:
            return

        selected_day = self.dropdown_days.value
        minutes = [
            (c.timestamp.strftime("%H:%M:%S"), index)
            for index, c in enumerate(self.clips)
            if c.timestamp.date() == selected_day
        ]
        minutes = sorted(minutes)

        if minutes:  # Only update if we have options
            self.selector_minutes.options = minutes
            self.selector_minutes.disabled = False
        else:
            self.selector_minutes.options = []
            self.selector_minutes.disabled = True

    def update_video(self, *args):
        """Queue a video clip to be displayed when minute is selected"""
        if self.selector_minutes.value is None:
            return

        selected_clip = self.clips[self.selector_minutes.value]
        self.clip_queue.put(selected_clip)

    def render(self):
        """Display the viewer interface"""
        return VBox(
            [
                self.dropdown_cameras,
                self.dropdown_days,
                self.selector_minutes,
                self.text_box,
                HBox([self.img_widget]),
            ]
        )

    @classmethod
    def create_from_path(cls, videos_path: Path | str) -> "JupyterViewer":
        """
        Create and display an Aqara video viewer

        Args:
            videos_path (str): Path to the directory containing video files

        Returns:
            AqaraViewer: The viewer instance
        """
        if isinstance(videos_path, str):
            videos_path = Path(videos_path)
        viewer = JupyterViewer(videos_path)
        display(viewer.render())
        return viewer
