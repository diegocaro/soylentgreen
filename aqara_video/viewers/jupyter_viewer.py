import threading
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import cv2
from IPython.display import display
from ipywidgets import Dropdown, HBox, Image, Layout, SelectionSlider, Text, VBox

from aqara_video.core.clip import Clip
from aqara_video.core.factory import TimelineFactory

MAPPING_CAMERAS = {
    "lumi1.54ef44457bc9": "Pasto",
    "lumi1.54ef44603857": "Living",
}

# Define callback types
ClipCallback = Callable[[Clip], None]
FrameCallback = Callable[[bytes, int], None]
FinishCallback = Callable[[], None]


class VideoProcessor:
    """Handles video loading and processing logic separate from the UI."""

    def __init__(
        self, videos_path: Path, cameras_names: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Video Processor

        Args:
            videos_path (Path): Path to the directory containing video files
            cameras_names (dict, optional): Mapping of camera names to human-readable names
        """
        self.videos_path = videos_path

        if cameras_names is None:
            cameras_names = MAPPING_CAMERAS
        self.mapping_cameras = cameras_names

        self.clips: List[Clip] = []
        self.clip_queue: Queue[Clip] = Queue(maxsize=1000)

        # Define callback handlers
        self.on_clip_loaded: Optional[ClipCallback] = None
        self.on_frame_ready: Optional[FrameCallback] = None
        self.on_clip_finished: Optional[FinishCallback] = None

        # Start video processing thread
        self.loop = threading.Thread(
            target=self.refresh_loop,
            daemon=True,
        )
        self.loop.start()

    @property
    def cameras(self) -> List[Tuple[str, Path]]:
        """Get a list of available cameras with their human-readable names."""
        return [
            (self.mapping_cameras.get(c.name, c.name), c)
            for c in Path(self.videos_path).glob("lumi1.*")
        ]

    def load_clips_for_camera(self, camera_path: Path) -> List[Clip]:
        """Load all clips for a specific camera."""
        self.clips = TimelineFactory.create_timeline(camera_path).clips
        return self.clips

    def get_unique_days(self) -> List[Any]:
        """Get the unique days for which clips are available."""
        return sorted(list(set(c.timestamp.date() for c in self.clips)))

    def get_clips_for_day(self, selected_day: Any) -> List[Tuple[str, int]]:
        """Get clips for a selected day formatted as (time_str, index)."""
        minutes = [
            (c.timestamp.strftime("%H:%M:%S"), index)
            for index, c in enumerate(self.clips)
            if c.timestamp.date() == selected_day
        ]
        return sorted(minutes)

    def queue_clip(self, clip_index: int) -> None:
        """Queue a clip to be processed."""
        if 0 <= clip_index < len(self.clips):
            self.clip_queue.put(self.clips[clip_index])

    def refresh_loop(self) -> None:
        """Thread function to process video clips and update the UI."""
        while True:
            selected_clip = self.clip_queue.get()
            while not self.clip_queue.empty():
                try:
                    selected_clip = self.clip_queue.get_nowait()
                except:
                    break

            # Signal to the UI that a new clip is being processed
            if self.on_clip_loaded is not None:
                self.on_clip_loaded(selected_clip)

            for frame_id, frame in selected_clip.frames():
                if not self.clip_queue.empty():
                    break  # New clip available, stop current playback

                if frame_id % 60 == 0:
                    _, jpeg = cv2.imencode(".jpg", frame)
                    jpeg_bytes = jpeg.tobytes()

                    # Signal to the UI that a new frame is available
                    if self.on_frame_ready is not None:
                        self.on_frame_ready(jpeg_bytes, frame_id)
                    break

            # Signal to advance to the next clip if available
            if self.on_clip_finished is not None:
                self.on_clip_finished()


class JupyterViewerUI:
    """Handles the UI components and user interaction for viewing videos in Jupyter."""

    def __init__(
        self, videos_path: Path, cameras_names: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Jupyter Viewer UI

        Args:
            videos_path (Path): Path to the directory containing video files
            cameras_names (dict, optional): Mapping of camera names to human-readable names
        """
        # Create the video processor
        self.processor = VideoProcessor(videos_path, cameras_names)

        # Set up callback handlers
        self.processor.on_clip_loaded = self.on_clip_loaded
        self.processor.on_frame_ready = self.on_frame_ready
        self.processor.on_clip_finished = self.on_clip_finished

        # Create widgets
        self.img_widget = Image()
        self.text_box = Text(description="File:", layout=Layout(width="1000px"))

        self.dropdown_cameras = Dropdown(
            options=self.processor.cameras,
            description="Camera:",
        )

        self.dropdown_days = Dropdown(description="Day:")
        self.selector_minutes = SelectionSlider(
            options=[("No videos available", "")],
            description="Minute:",
            disabled=True,
            layout=Layout(width="1000px"),
        )

        # Set up event handlers
        self.dropdown_cameras.observe(self.update_days, names="value")
        self.dropdown_days.observe(self.update_minutes, names="value")
        self.selector_minutes.observe(self.update_video, names="value")

        # Initial update
        self.update_days()

    def update_days(self, *args: Any) -> None:
        """Update the days dropdown based on camera selection."""
        self.processor.load_clips_for_camera(self.dropdown_cameras.value)

        days = self.processor.get_unique_days()
        self.dropdown_days.options = days

        # Also update minutes when days change
        self.update_minutes()

    def update_minutes(self, *args: Any) -> None:
        """Update the minutes selection slider based on day selection."""
        if self.dropdown_days.value is None:
            return

        minutes = self.processor.get_clips_for_day(self.dropdown_days.value)

        if minutes:  # Only update if we have options
            self.selector_minutes.options = minutes
            self.selector_minutes.disabled = False
        else:
            self.selector_minutes.options = []
            self.selector_minutes.disabled = True

    def update_video(self, *args: Any) -> None:
        """Queue a video clip to be displayed when minute is selected."""
        if self.selector_minutes.value is None:
            return

        self.processor.queue_clip(self.selector_minutes.value)

    def on_clip_loaded(self, clip: Clip) -> None:
        """Callback for when a clip begins processing."""
        self.text_box.value = (
            f"{clip.path} fps={clip.fps:.2f}, {clip.width}x{clip.height}"
        )

    def on_frame_ready(self, jpeg_bytes: bytes, frame_id: int) -> None:
        """Callback for when a new frame is ready to display."""
        self.img_widget.value = jpeg_bytes

    def on_clip_finished(self) -> None:
        """Callback for when clip playback is complete."""
        # Advance to the next clip if available (same behavior as before)
        if self.selector_minutes.index + 1 < len(self.selector_minutes.options):
            self.selector_minutes.index += 1

    def render(self):
        """Display the viewer interface."""
        return VBox(
            [
                self.dropdown_cameras,
                self.dropdown_days,
                self.selector_minutes,
                self.text_box,
                HBox([self.img_widget]),
            ]
        )


class JupyterViewer:
    """Legacy class that maintains backwards compatibility with existing code."""

    def __init__(
        self, videos_path: Path, cameras_names: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Aqara Viewer

        Args:
            videos_path (Path): Path to the directory containing video files
            cameras_names (dict, optional): Mapping of camera names to human-readable names
        """
        self.ui = JupyterViewerUI(videos_path, cameras_names)
        self.processor = self.ui.processor

        # Mirror properties from the UI for backward compatibility
        self.videos_path = videos_path
        self.mapping_cameras = self.processor.mapping_cameras
        self.img_widget = self.ui.img_widget
        self.dropdown_cameras = self.ui.dropdown_cameras
        self.text_box = self.ui.text_box
        self.dropdown_days = self.ui.dropdown_days
        self.selector_minutes = self.ui.selector_minutes
        self.clips = self.processor.clips
        self.clip_queue = self.processor.clip_queue

    @property
    def cameras(self):
        return self.processor.cameras

    def update_days(self, *args: Any) -> None:
        self.ui.update_days(*args)

    def update_minutes(self, *args: Any) -> None:
        self.ui.update_minutes(*args)

    def update_video(self, *args: Any) -> None:
        self.ui.update_video(*args)

    def render(self):
        return self.ui.render()

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
