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
    QHBoxLayout,
    QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont


class CustomTimeline(QWidget):
    timeSelected = pyqtSignal(datetime.datetime)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.setMaximumHeight(100)

        self.videos = {}  # {datetime: filepath}
        self.start_date = None
        self.end_date = None
        self.current_time = None
        self.setMouseTracking(True)

    def set_data(self, videos, start_date, end_date):
        self.videos = videos
        self.start_date = start_date
        self.end_date = end_date
        self.update()

    def paintEvent(self, event):
        if not self.start_date or not self.end_date:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(event.rect(), QColor(40, 40, 40))

        width = self.width()
        height = self.height()
        total_seconds = (self.end_date - self.start_date).total_seconds()

        # Draw hour markers
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        current_time = self.start_date
        while current_time <= self.end_date:
            x = int(
                ((current_time - self.start_date).total_seconds() / total_seconds)
                * width
            )

            # Major markers for days
            if current_time.hour == 0:
                painter.setPen(QPen(QColor(200, 200, 200), 2))
                painter.drawLine(x, 0, x, height)
                date_str = current_time.strftime("%Y-%m-%d")
                painter.drawText(x + 5, 15, date_str)
            # Minor markers for hours
            elif current_time.hour % 6 == 0:
                painter.setPen(QPen(QColor(100, 100, 100), 1))
                painter.drawLine(x, height - 20, x, height)
                painter.drawText(x - 10, height - 5, f"{current_time.hour:02d}:00")

            current_time += datetime.timedelta(hours=1)

        # Draw video segments
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        for video_time in self.videos.keys():
            x = int(
                ((video_time - self.start_date).total_seconds() / total_seconds) * width
            )
            painter.drawLine(x, 20, x, height - 25)

        # Draw current position
        if self.current_time:
            x = int(
                ((self.current_time - self.start_date).total_seconds() / total_seconds)
                * width
            )
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.drawLine(x, 0, x, height)

    def mousePressEvent(self, event):
        if not self.start_date or not self.end_date:
            return

        width = self.width()
        x = event.position().x()
        total_seconds = (self.end_date - self.start_date).total_seconds()
        target_seconds = (x / width) * total_seconds
        target_time = self.start_date + datetime.timedelta(seconds=target_seconds)

        # Find nearest video
        nearest_time = min(
            self.videos.keys(), key=lambda t: abs((t - target_time).total_seconds())
        )
        self.current_time = nearest_time
        self.timeSelected.emit(nearest_time)
        self.update()


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

        # Add controls layout
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        # controls_layout.addWidget(self.play_button)

        # Add controls to main layout
        layout.addLayout(controls_layout)

        # Replace timeline slider with custom timeline
        self.timeline = CustomTimeline()
        self.timeline.timeSelected.connect(self.timeline_changed)
        layout.addWidget(self.timeline)

        # Video playback variables
        self.is_playing = False
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
            self.timeline.set_data(self.video_files, self.start_date, self.end_date)
            print(f"Found videos from {self.start_date} to {self.end_date}")

    def timeline_changed(self, selected_time):
        if selected_time in self.video_files:
            self.load_video(self.video_files[selected_time])

    def load_video(self, video_path):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        self.current_video = video_path

        # Start playing when a new video is loaded
        if not self.is_playing:
            self.toggle_playback()

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

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_button.setText("Pause" if self.is_playing else "Play")

        if self.is_playing:
            self.timer.start(33)  # ~30 fps
        else:
            self.timer.stop()

    def update_frame(self):
        if not self.is_playing or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            # Video ended, move to next available video
            current_time = None
            for video_time in sorted(self.video_files.keys()):
                if Path(self.video_files[video_time]) == Path(self.current_video):
                    # Find next video
                    times = sorted(self.video_files.keys())
                    current_idx = times.index(video_time)
                    if current_idx + 1 < len(times):
                        current_time = times[current_idx + 1]
                    break

            if current_time:
                self.timeline.current_time = current_time
                self.load_video(self.video_files[current_time])
                self.timeline.update()
            else:
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
