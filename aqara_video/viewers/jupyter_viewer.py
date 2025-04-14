import threading
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from IPython.display import display
from ipywidgets import (
    Button,
    Checkbox,
    Dropdown,
    HBox,
    Image,
    Layout,
    SelectionSlider,
    Text,
    VBox,
)

from aqara_video.core.clip import Clip
from aqara_video.core.factory import TimelineFactory
from aqara_video.core.images import encode_to_jpeg
from aqara_video.ml.detector import Detector
from aqara_video.ml.utils import draw_boxes
from aqara_video.providers.aqara import AqaraProvider

MAPPING_CAMERAS = {
    "lumi1.54ef44457bc9": "Pasto",
    "lumi1.54ef44603857": "Living",
}

# Define simple callback types
ClipCallback = Callable[[Clip], None]
FrameCallback = Callable[[Clip, bytes, int], None]
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
        self.first_frame_only: bool = True  # Default to showing only first frame
        self.paused: bool = True  # Start in paused state
        self.current_clip_index: Optional[int] = None  # Track current clip index
        self.resume_playback: bool = False

        # Object detection settings
        self.detect_objects: bool = True  # Default to detecting objects
        self.detection_threshold: float = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = None  # Lazy initialization for detector

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
            (self.mapping_cameras.get(c, c), self.videos_path / c)
            for c in AqaraProvider.cameras_in_dir(self.videos_path)
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
            self.current_clip_index = clip_index
            self.clip_queue.put(self.clips[clip_index])

    def toggle_pause(self) -> bool:
        """Toggle the pause state of the video playback.

        Returns:
            bool: The new pause state (True = paused, False = playing)
        """
        self.paused = not self.paused

        # If we're resuming playback, set the resume flag but don't queue
        # a new clip - instead, we'll continue from where we left off
        if not self.paused and self.current_clip_index is not None:
            self.resume_playback = True
            self.clip_queue.put(self.clips[self.current_clip_index])

        return self.paused

    def previous_clip(self) -> None:
        """Load the previous clip if available."""
        if self.current_clip_index is not None and self.current_clip_index > 0:
            self.queue_clip(self.current_clip_index - 1)

    def next_clip(self) -> None:
        """Load the next clip if available."""
        if (
            self.current_clip_index is not None
            and self.current_clip_index < len(self.clips) - 1
        ):
            self.queue_clip(self.current_clip_index + 1)

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

            # Reset current frame when loading a new clip, unless we're resuming playback
            if not self.resume_playback:
                # play from the start
                pass
            self.resume_playback = False  # Reset the resume flag

            # Initialize detector if needed and object detection is enabled
            if self.detect_objects and self.detector is None:
                self.detector = Detector(device=self.device, batch_size=1)

            # Process the clip's frames
            for frame in selected_clip.frames():
                if not self.clip_queue.empty():
                    break  # New clip available, stop current playback

                if self.paused and frame.frame_id > 0:
                    break  # Stop if paused, but always show at least the first frame

                # Apply object detection if enabled
                current_frame = frame.frame
                if self.detect_objects and self.detector is not None:
                    # Preprocess the frame
                    tensor = self.detector.preprocess(current_frame)
                    # Make predictions
                    predictions = self.detector.predict(tensor)
                    # Draw bounding boxes
                    current_frame = draw_boxes(
                        current_frame.copy(),
                        predictions,
                        threshold=self.detection_threshold,
                    )

                # Encode the frame as JPEG
                jpeg_bytes = encode_to_jpeg(current_frame)

                # Signal to the UI that a new frame is available
                if self.on_frame_ready is not None:
                    self.on_frame_ready(selected_clip, jpeg_bytes, frame.frame_id)

                if self.first_frame_only and frame.frame_id == 0:
                    break

            # Signal to advance to the next clip if available
            if self.on_clip_finished is not None and not self.paused:
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

        # Create first frame toggle checkbox
        self.first_frame_checkbox = Checkbox(
            value=self.processor.first_frame_only,
            description="Show only first frame",
            disabled=False,
        )

        # Create object detection checkbox
        self.detect_objects_checkbox = Checkbox(
            value=self.processor.detect_objects,
            description="Detect objects",
            disabled=False,
        )

        # Create playback control buttons
        self.play_button = Button(
            description="Play",
            disabled=False,
            button_style="success",
            tooltip="Play/Stop video",
            icon="play",
        )

        self.prev_button = Button(
            description="Previous",
            disabled=False,
            button_style="info",
            tooltip="Previous clip",
            icon="step-backward",
        )

        self.next_button = Button(
            description="Next",
            disabled=False,
            button_style="info",
            tooltip="Next clip",
            icon="step-forward",
        )

        # Set up button callback handlers
        self.play_button.on_click(self.on_play_button_click)
        self.prev_button.on_click(self.on_prev_button_click)
        self.next_button.on_click(self.on_next_button_click)

        # Controls container
        self.controls = HBox([self.prev_button, self.play_button, self.next_button])

        # Set up event handlers
        self.dropdown_cameras.observe(self.update_days, names="value")
        self.dropdown_days.observe(self.update_minutes, names="value")
        self.selector_minutes.observe(self.update_video, names="value")
        self.first_frame_checkbox.observe(self.toggle_first_frame_only, names="value")
        self.detect_objects_checkbox.observe(
            self.toggle_object_detection, names="value"
        )

        # Initial update
        self.update_days()

    def update_days(self, *args: Any) -> None:
        """Update the days dropdown based on camera selection."""
        if not self.dropdown_cameras.value:
            return

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

            if self.processor.paused:
                self.on_play_button_click(None)  # type: ignore
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
            f"{clip.path} fps={clip.fps:.2f}, {clip.width}x{clip.height}, frame_id=0"
        )

    def on_frame_ready(self, clip: Clip, jpeg_bytes: bytes, frame_id: int) -> None:
        """Callback for when a new frame is ready to display."""
        self.img_widget.value = jpeg_bytes
        self.text_box.value = f"{clip.path} fps={clip.fps:.2f}, {clip.width}x{clip.height}, frame_id={frame_id}"

    def on_clip_finished(self) -> None:
        """Callback for when clip playback is complete."""
        # Advance to the next clip if available
        if self.selector_minutes.index + 1 < len(self.selector_minutes.options):
            self.selector_minutes.index += 1

    def toggle_first_frame_only(self, change: Dict[str, Any]) -> None:
        """Toggle the first frame only setting in the processor."""
        self.processor.first_frame_only = change["new"]

        # If we're switching modes, reload the current clip to see the effect immediately
        if self.selector_minutes.value is not None:
            self.processor.queue_clip(self.selector_minutes.value)

    def toggle_object_detection(self, change: Dict[str, Any]) -> None:
        """Toggle the object detection setting in the processor."""
        self.processor.detect_objects = change["new"]

        # If we're switching modes, reload the current clip to see the effect immediately
        if self.selector_minutes.value is not None:
            self.processor.queue_clip(self.selector_minutes.value)

    def on_play_button_click(self, b: Button) -> None:
        """Callback for when the play button is clicked."""
        self.processor.toggle_pause()
        self.play_button.icon = "pause" if not self.processor.paused else "play"
        self.play_button.description = "Pause" if not self.processor.paused else "Play"

    def on_prev_button_click(self, b: Button) -> None:
        """Callback for when the previous button is clicked."""
        self.processor.previous_clip()

    def on_next_button_click(self, b: Button) -> None:
        """Callback for when the next button is clicked."""
        self.processor.next_clip()

    def render(self):
        """Display the viewer interface."""
        return VBox(
            [
                HBox(
                    [self.dropdown_cameras, self.dropdown_days, self.selector_minutes]
                ),
                self.text_box,
                HBox(
                    [
                        self.controls,
                        self.first_frame_checkbox,
                        self.detect_objects_checkbox,
                    ]
                ),
                HBox([self.img_widget]),
            ]
        )

    @classmethod
    def create_from_path(cls, videos_path: Union[Path, str]) -> "JupyterViewerUI":
        """
        Create and display a Jupyter viewer widget

        Args:
            videos_path (str or Path): Path to the directory containing video files

        Returns:
            JupyterViewerUI: The viewer instance
        """
        if isinstance(videos_path, str):
            videos_path = Path(videos_path)

        viewer = JupyterViewerUI(videos_path)
        display(viewer.render())
        return viewer
