# Video Processing API

## VideoProcessor

The main class for processing video streams and files.

### Class Definition

```python
from earsegmentationai import VideoProcessor
```

### Constructor

```python
VideoProcessor(
    device: str = "cpu",
    threshold: float = 0.5,
    skip_frames: int = 0,
    smooth_masks: bool = False,
    temporal_window: int = 5
)
```

**Parameters:**
- `device` (str): Processing device ("cpu" or "cuda:0")
- `threshold` (float): Binary threshold for mask generation (0.0 to 1.0)
- `skip_frames` (int): Number of frames to skip between processing
- `smooth_masks` (bool): Apply temporal smoothing to masks
- `temporal_window` (int): Window size for temporal smoothing

### Methods

#### process()

Process video file or camera stream.

```python
def process(
    input_source: Union[str, Path, int],
    output_path: Optional[Union[str, Path]] = None,
    display: bool = False,
    save_masks: bool = False,
    mask_dir: Optional[Union[str, Path]] = None,
    max_frames: Optional[int] = None
) -> VideoStats
```

**Parameters:**
- `input_source`: Input video source:
  - Path to video file
  - Camera device ID (0 for default camera)
  - URL for network stream
- `output_path` (Optional): Path to save output video
- `display` (bool): Display real-time preview
- `save_masks` (bool): Save individual frame masks
- `mask_dir` (Optional): Directory for saving masks
- `max_frames` (Optional): Maximum frames to process

**Returns:**
- `VideoStats`: Processing statistics

### VideoStats

Statistics from video processing.

**Attributes:**
- `total_frames` (int): Total frames processed
- `frames_with_ears` (int): Frames where ears were detected
- `detection_rate` (float): Percentage of frames with ears
- `average_fps` (float): Average processing FPS
- `total_time` (float): Total processing time
- `output_path` (Optional[str]): Path to output video
- `frame_dimensions` (Tuple[int, int]): Video dimensions (width, height)

### CameraProcessor

Specialized class for real-time camera processing.

```python
from earsegmentationai import CameraProcessor

processor = CameraProcessor(
    device="cuda:0",
    display_fps: bool = True,
    record: bool = False
)
```

### Methods

#### start()

Start camera processing.

```python
def start(
    camera_id: int = 0,
    output_path: Optional[str] = None,
    duration: Optional[int] = None
) -> CameraStats
```

**Parameters:**
- `camera_id` (int): Camera device ID
- `output_path` (Optional[str]): Path to save recording
- `duration` (Optional[int]): Recording duration in seconds

#### stop()

Stop camera processing.

```python
def stop() -> None
```

## Examples

### Basic Video Processing

```python
from earsegmentationai import VideoProcessor

# Initialize processor
processor = VideoProcessor(device="cuda:0")

# Process video file
stats = processor.process(
    "input_video.mp4",
    output_path="output_video.mp4"
)

print(f"Processed {stats.total_frames} frames")
print(f"Detection rate: {stats.detection_rate:.1f}%")
print(f"Average FPS: {stats.average_fps:.1f}")
```

### Real-time Camera Processing

```python
from earsegmentationai import CameraProcessor

# Initialize camera processor
processor = CameraProcessor(
    device="cuda:0",
    display_fps=True
)

# Start processing (press 'q' to stop)
stats = processor.start(
    camera_id=0,
    output_path="recording.mp4"
)
```

### Advanced Options

```python
# Process with frame skipping for better performance
processor = VideoProcessor(
    device="cuda:0",
    skip_frames=2,  # Process every 3rd frame
    smooth_masks=True,  # Temporal smoothing
    temporal_window=7
)

# Process with all options
stats = processor.process(
    "video.mp4",
    output_path="output.mp4",
    display=True,
    save_masks=True,
    mask_dir="masks/",
    max_frames=1000
)
```

### Network Stream Processing

```python
# Process RTSP stream
processor = VideoProcessor(device="cuda:0")
stats = processor.process(
    "rtsp://192.168.1.100:554/stream",
    output_path="stream_output.mp4"
)

# Process HTTP stream
stats = processor.process(
    "http://example.com/stream.m3u8",
    display=True
)
```

### Callback Integration

```python
# Custom callback for each frame
def frame_callback(frame_num, has_ear, confidence):
    print(f"Frame {frame_num}: Ear={'Yes' if has_ear else 'No'}, Conf={confidence:.2f}")

processor = VideoProcessor(device="cuda:0")
processor.set_callback(frame_callback)
stats = processor.process("video.mp4")
```

### Error Handling

```python
try:
    stats = processor.process("video.mp4")
except FileNotFoundError:
    print("Video file not found")
except PermissionError:
    print("Cannot access camera")
except Exception as e:
    print(f"Processing error: {e}")
```