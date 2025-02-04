import sys
import cv2
import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


class VideoViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing VideoViewer...")
        self.setWindowTitle("Camera Video Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Video display
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # Timeline slider
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(24 * 60)  # Minutes in a day
        self.timeline.valueChanged.connect(self.timeline_changed)
        layout.addWidget(self.timeline)

        # Video playback variables
        self.video_path = "/Volumes/Cameras/aqara_video/lumi1.54ef44457bc9"
        if not Path(self.video_path).exists():
            print(f"Error: Video directory not found: {self.video_path}")
            self.video_path = "."  # Fallback to current directory

        self.cap = None
        self.current_video = None
        self.video_files = {}
        print("Scanning for video files...")
        self.scan_video_files()
        print(f"Found {len(self.video_files)} video files")

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 fps

    def scan_video_files(self):
        base_path = Path(self.video_path)
        for day_dir in base_path.glob("*"):
            if not day_dir.is_dir() or not day_dir.name.isdigit():
                continue

            for video_file in day_dir.glob("*.mp4"):
                time_str = video_file.stem
                if len(time_str) != 6:
                    continue

                try:
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])
                    minutes_of_day = hour * 60 + minute
                    self.video_files[minutes_of_day] = str(video_file)
                except ValueError:
                    continue

    def timeline_changed(self, value):
        target_time = value
        if target_time in self.video_files:
            self.load_video(self.video_files[target_time])

    def load_video(self, video_path):
        try:
            if self.cap is not None:
                self.cap.release()

            print(f"Loading video: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return
            self.current_video = video_path
        except Exception as e:
            print(f"Error loading video: {e}")

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            # Video ended, reload it
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Convert frame to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale the image to fit the label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(scaled_pixmap)


def main():
    try:
        app = QApplication(sys.argv)
        print("Starting Video Viewer application...")
        viewer = VideoViewer()
        viewer.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
