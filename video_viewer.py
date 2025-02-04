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
        self.cap = None
        self.current_video = None
        self.video_files = {}  # Changed to {datetime: filepath}
        self.start_date = None
        self.end_date = None
        self.scan_video_files()

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 fps

    def scan_video_files(self):
        base_path = Path(self.video_path)
        self.video_files.clear()

        for day_dir in base_path.glob("*"):
            if not day_dir.is_dir() or not day_dir.name.isdigit():
                continue

            try:
                day = datetime.datetime.strptime(day_dir.name, "%Y%m%d")

                for video_file in day_dir.glob("*.mp4"):
                    time_str = video_file.stem
                    if len(time_str) != 6:
                        continue

                    try:
                        hour = int(time_str[:2])
                        minute = int(time_str[2:4])
                        second = int(time_str[4:6])

                        video_datetime = day.replace(
                            hour=hour, minute=minute, second=second
                        )
                        self.video_files[video_datetime] = str(video_file)
                    except ValueError:
                        continue
            except ValueError:
                continue

        if self.video_files:
            self.start_date = min(self.video_files.keys())
            self.end_date = max(self.video_files.keys())
            print(f"Found videos from {self.start_date} to {self.end_date}")

    def timeline_changed(self, value):
        if not self.video_files or not self.start_date or not self.end_date:
            return

        # Convert slider value (0-1440) to datetime
        total_minutes = (self.end_date - self.start_date).total_seconds() / 60
        target_minutes = value
        target_time = self.start_date + datetime.timedelta(minutes=target_minutes)

        # Find nearest video
        nearest_time = min(
            self.video_files.keys(),
            key=lambda x: abs((x - target_time).total_seconds()),
        )
        self.load_video(self.video_files[nearest_time])

    def load_video(self, video_path):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        self.current_video = video_path

        # After loading video, update window title with current video datetime
        if self.current_video:
            video_path = Path(self.current_video)
            day_str = video_path.parent.name
            time_str = video_path.stem
            try:
                date = datetime.datetime.strptime(
                    f"{day_str}{time_str}", "%Y%m%d%H%M%S"
                )
                self.setWindowTitle(
                    f"Camera Video Viewer - {date.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            except ValueError:
                self.setWindowTitle("Camera Video Viewer")

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
    app = QApplication(sys.argv)
    viewer = VideoViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
