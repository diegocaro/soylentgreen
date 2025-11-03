# SoylentGreen - WIP

Tools for creating timelapse videos from surveillance footage, with a focus on Aqara security cameras.

(or how we discovered that a plant can be killed by the sun)

![Lavanda killed by the sun](assets/lavanda.gif)

![Wheatgrass growing up! - some mold spotted](assets/pasto.gif)


## Goal (for me)

Learn about computational vision and machine learning.

## Overview

SoylentGreen is a Python package that provides tools to work with surveillance camera footage, particularly from Aqara security cameras. It allows you to:

- Create timelapses from surveillance footage
- View videos within Jupyter notebooks
- Process and analyze video frames
- Apply intelligent filtering based on color analysis
- Run and train machine learning algorithms


## File Storage Structure

Aqara security cameras save video files in a structured directory format:

```
/path/to/aqara_video/[camera_id]/[YYYYMMDD]/[HHMMSS].mp4
```

Where:
- **camera_id**: A unique identifier for each camera with format `lumi1.[alphanumeric_string]` (e.g., `lumi1.54ef44457bc9`)
- **YYYYMMDD**: Date folder organized by year, month, and day (e.g., `20250207` for February 7, 2025)
- **HHMMSS**: Time-based filename with hours, minutes, and seconds (e.g., `082900.mp4` for 8:29:00 AM)

### NAS Integration via Samba

You can configure Aqara cameras to save footage directly to a NAS through the iOS/Android Aqara app. Once configured, you can store and access Aqara camera footage on a Network Attached Storage (NAS) using Samba shares.

For example, if your NAS is mounted at `/mnt/cameras/`, a complete video path might look like:
```
/mnt/cameras/aqara_video/lumi1.54ef44457bc9/20250301/185800.mp4
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [pre-commit](https://pre-commit.com/) for code quality checks.

```bash
# Clone the repository
git clone https://github.com/diegocaro/soylentgreen.git
cd soylentgreen

# Install dependencies with uv (using pyproject.toml)
uv sync

# (Optional) Set up pre-commit hooks for linting and type checking
pre-commit install
```


## Dependencies

- Python 3.12+
- OpenCV
- FFmpeg
- NumPy
- Joblib
- Pytest
- [uv](https://github.com/astral-sh/uv) (dependency management, uses `pyproject.toml`)
- [pre-commit](https://pre-commit.com/) (git hooks)
- [pyright](https://github.com/microsoft/pyright) (type checking)
- [ruff](https://github.com/astral-sh/ruff) (linting)


## Development Workflow

Before committing code, pre-commit will automatically run [pyright](https://github.com/microsoft/pyright) for type checking and [ruff](https://github.com/astral-sh/ruff) for linting. You can run these checks manually:

```bash
pre-commit run --all-files
```

## Usage

### Creating Timelapses

```bash
# Basic timelapse from a camera directory
timelapse /path/to/aqara_video/lumi1.54ef44457bc9 --output output.mp4

# Timelapse for a specific date range
timelapse /path/to/aqara_video/lumi1.54ef44457bc9 --date-from 20250301-000000 --date-to 20250301-235959 --output output.mp4

# Timelapse for a specific day with green filtering (for plants/gardens)
timelapse /path/to/aqara_video/lumi1.54ef44457bc9 --day 20250301 --green-threshold 0.3 --output output.mp4
```

### Using in Jupyter Notebooks
The JupyterViewer widget shows a list of videos recorded per day.

```python
from aqara_video.viewers.jupyter_viewer import JupyterViewer

# Create and display a viewer for browsing camera footage
viewer = JupyterViewer.create_from_path("/path/to/aqara_video/lumi1.54ef44457bc9")
```
![JupyterViewer widget](assets/jupyter_viewer2.png)

### Video Analysis

```python
from pathlib import Path
from aqara_video.core.factory import TimelineFactory
from aqara_video.core.video_reader import VideoReader

# Create a timeline of clips from a camera
timeline = TimelineFactory.create_timeline(Path("/path/to/camera/directory"))

# Read the first frame from a video file
video = VideoReader(Path("/path/to/video.mp4"))
frame = video.read_frame()
```

## Project Structure

- `aqara_video/`: Main package
  - `cli/`: Command-line tools
    - `timelapse.py`: Create timelapse videos
    - `video_loop.py`: Real-time video processing
  - `core/`: Core functionality
    - `clip.py`: Video clip representation
    - `factory.py`: Factory for creating timelines
    - `timeline.py`: Timeline representation
    - `video_reader.py`: Low-level video reading tools
  - `providers/`: Camera providers
    - `aqara.py`: Support for Aqara cameras
  - `viewers/`: Video viewers
    - `jupyter_viewer.py`: Jupyter notebook integration

## Examples

Check the `examples/` directory for Jupyter notebooks and Python scripts demonstrating usage.

## License

[License information]

## Author

Diego Caro