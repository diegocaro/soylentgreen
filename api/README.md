# NAS Video Timeline Viewer

A web application for browsing, searching, and playing back video recordings from a NAS (Network Attached Storage) camera system with intelligent object detection overlays.

## Overview

This application provides a sophisticated interface for navigating through camera recordings stored on a NAS. It displays video availability timelines, detected object labels (boxes, persons, cars), and allows seamless playback across multiple video files with automatic transitions.

## Key Features

### 1. Multi-Camera Support
- **Camera Selection**: Dropdown selector to switch between different cameras
- **Dynamic Loading**: Cameras are loaded from the backend API at startup
- **Per-Camera Data**: Each camera has its own video intervals and label detection data

### 2. Dual Timeline Interface

#### Overview Timeline (Top)
- **Purpose**: High-level view of all available recordings across days/weeks
- **Zoom Range**: 1 day minimum to 1 year maximum
- **Visual Indicators**:
  - Green gradient bars show video availability
  - Colored overlays show detected label intervals
  - Blue range indicator shows the currently visible detailed view range
  - Red playback indicator shows current video playback position
- **Interaction**: Click anywhere to jump to that date/time and start playing video

#### Detailed Timeline (Bottom)
- **Purpose**: Fine-grained view for precise navigation within hours
- **Zoom Range**: 5 minutes minimum to 24 hours maximum
- **Visual Indicators**:
  - Video availability intervals
  - Label detection overlays with color coding
  - Current time indicator
  - Red playback indicator during video playback
- **Interaction**: Click to seek to specific timestamp
- **Hover**: Shows scrub line and time tooltip for preview

### 3. Label Detection Visualization

The app displays object detection results overlaid on the timelines:

- **Supported Labels**: 
  - `box` - Orange (rgba(255, 152, 0, 0.3))
  - `person` - Blue (rgba(33, 150, 243, 0.3))
  - `car` - Red (rgba(244, 67, 54, 0.3))
  - `default` - Purple (rgba(156, 39, 176, 0.3))

- **Label Legend**: Below the detailed timeline, shows all detected labels with color indicators
- **Time Intervals**: Each label type shows time ranges when objects were detected
- **Dynamic Rendering**: Labels re-render as you zoom/pan the timelines
- **Tooltips**: Hover over label markers to see label name and time range

### 4. Advanced Video Playback

#### Dual-Buffer System
- **Two Video Players**: playerA and playerB for seamless transitions
- **Z-index Swapping**: Active player on top, inactive below
- **Purpose**: Eliminates loading delays between video files

#### Automatic Playback Features
- **Continuous Play**: Automatically advances to the next video when current ends
- **Preloading**: Next video is preloaded in the background while current video plays
- **Smart Seeking**: Calculates exact timestamps across video boundaries
- **Seamless Transitions**: Instant switch between videos with pre-seeked position

#### Playback Indicators
- **Red Line**: Shows current playback position on both timelines
- **Real-time Sync**: Updates 10 times per second (100ms interval)
- **Accurate Timestamps**: Calculates actual time based on seek position and video offset

### 5. Interactive Features

#### Timeline Scrubbing
- **Mouse Hover**: Shows vertical scrub line and time tooltip
- **Both Timelines**: Works on overview and detailed views
- **Time Format**: 
  - Overview: Date + HH:MM:SS
  - Detailed: HH:MM:SS

#### Video Seeking
- **Click to Seek**: Click on any timeline to jump to that timestamp
- **Backend Integration**: Calls `/seek` API to find correct video file and offset
- **Auto-play**: Automatically starts playing after seeking

#### Range Synchronization
- **Blue Box Indicator**: Shows detailed view range on overview timeline
- **Dynamic Updates**: Updates as you zoom/pan the detailed timeline
- **Visual Feedback**: Helps maintain context while navigating

### 6. Timeline Interactions

#### Zooming
- **Mouse Wheel**: Zoom in/out on timelines
- **Pinch Gesture**: Touch device support
- **Constraints**: 
  - Overview: 1 day to 1 year
  - Detailed: 5 minutes to 24 hours

#### Panning
- **Click and Drag**: Move through time
- **Synchronized**: Overview and detail indicators stay in sync

## Technical Implementation

### Frontend Architecture

#### Libraries
- **vis-timeline**: Powers the interactive timeline components (v7.7.0)
- **Native HTML5**: Video playback using `<video>` elements

#### State Management
- `window._sg_timeline`: Detailed timeline instance
- `window._sg_overviewTimeline`: Overview timeline instance
- `window._sg_intervals`: Video availability intervals
- `window._sg_labels`: Label detection data
- `window._sg_playbackInterval`: Playback position update timer

#### Video Player Management
- **Active Player**: Currently visible and playing video
- **Inactive Player**: Background player for preloading
- **Metadata Tracking**: 
  - `seekTimestamp`: Original seek time
  - `videoOffset`: Start position in video file
  - `videoDuration`: Total video length
  - `preloadedOffset`: Preloaded seek position

### Backend API Endpoints

1. **GET /cameras**
   - Returns list of available cameras
   - Response: `[{id: string, name: string}, ...]`

2. **GET /list-intervals?camera_id={id}**
   - Returns video availability intervals for a camera
   - Response: `[{start: ISO8601, end: ISO8601}, ...]`

3. **GET /labels?camera_id={id}**
   - Returns object detection labels timeline
   - Response: `{labels: {label_name: {intervals: [{start, end}, ...]}}}`

4. **GET /seek?time={ISO8601}&camera_id={id}**
   - Finds video file containing the requested timestamp
   - Response: `{path: string, offset: number}` (offset in seconds)

5. **GET /video?path={path}**
   - Streams video file for playback
   - Returns video file with appropriate headers

### Data Flow

1. **App Initialization**:
   - Load cameras from `/cameras`
   - Select first camera by default
   - Initialize timelines for selected camera

2. **Timeline Loading**:
   - Fetch intervals from `/list-intervals`
   - Fetch labels from `/labels`
   - Render both timelines with intervals
   - Overlay label markers on timelines
   - Display label legend

3. **Video Playback**:
   - User clicks timeline
   - Call `/seek` to get video file and offset
   - Load video in inactive player
   - Seek to offset when metadata loads
   - Start playback and swap players
   - Preload next video in background

4. **Auto-Advance**:
   - Video fires `ended` event
   - Check if next video is preloaded
   - If yes: swap and play immediately
   - If no: calculate next timestamp, call `/seek`, load and play

## File Structure

```
api/
├── index.html           # Frontend application (this file)
├── app.py              # FastAPI backend server
├── service.py          # Business logic for video/label management
├── models.py           # Pydantic data models
├── config.py           # Configuration (paths, camera mapping)
├── scan_result.json    # Scanned video intervals
├── labels_timeline.json # Object detection results
└── README.md           # This file
```

## Usage

1. **Start the Application**:
   - Run the FastAPI server (typically with `uvicorn app:app`)
   - Open browser to http://localhost:8000

2. **Select Camera**:
   - Use dropdown in top-right to switch cameras

3. **Navigate**:
   - Use overview timeline for broad jumps (days/weeks)
   - Use detailed timeline for precise navigation (minutes/hours)
   - Zoom with mouse wheel, pan by dragging

4. **Play Video**:
   - Click on either timeline to seek and play
   - Video plays automatically and continues across files
   - Use browser video controls (play/pause/volume)

5. **View Detections**:
   - Colored overlays show when objects were detected
   - Check legend below timeline for label meanings
   - Hover over overlays to see exact time ranges

## Performance Optimizations

1. **Dual Video Buffers**: Eliminates loading gaps between videos
2. **Preloading**: Next video loads while current plays
3. **Selective Rendering**: Labels only render for visible time range
4. **Efficient Updates**: Playback indicator updates at 10Hz (not every frame)
5. **Conditional Display**: Indicators hide when outside visible range

## Future Enhancement Ideas

- Video thumbnails on hover
- Multi-camera synchronized playback
- Download/export video segments
- Search by detected labels
- Playback speed control
- Mobile-optimized interface
- Live camera feed integration
