# Frame Capture Tool

A simple Python script to capture frames from video sources (webcam or video files).

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)

## Usage

### Capture from Webcam
```bash
python frame_capture.py --source 0 --output webcam_frame.jpg
```

### Capture from Video File
```bash
python frame_capture.py --source path/to/video.mp4 --output video_frame.jpg
```

### Capture Specific Frame from Video File
```bash
python frame_capture.py --source path/to/video.mp4 --frame 100 --output frame_100.jpg
```

## Arguments

- `--source`: Video source (use 0 for default webcam, or path to video file)
- `--frame`: Frame number to capture (optional, only works with video files)
- `--output`: Output path for the captured frame (default: captured_frame.jpg)
