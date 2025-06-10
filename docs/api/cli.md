# CLI Reference

The Ear Segmentation AI command-line interface provides easy access to all features.

## Installation

After installing the package, the `earsegmentationai` command will be available:

```bash
earsegmentationai --help
```

## Commands

### version

Display version information.

```bash
earsegmentationai version
```

**Output:**
```
Ear Segmentation AI v2.0.0
Python: 3.8.0
PyTorch: 1.13.0
Device: cuda (NVIDIA GeForce RTX 3080)
Model: earsegmentation_model_v1_46.pth
```

### process-image

Process static images for ear segmentation.

```bash
earsegmentationai process-image [OPTIONS] PATH
```

**Arguments:**
- `PATH`: Path to image file or directory

**Options:**
- `--output, -o PATH`: Output directory for results
- `--device, -d TEXT`: Processing device (cpu, cuda:0) [default: cpu]
- `--threshold, -t FLOAT`: Binary threshold [0.0-1.0] [default: 0.5]
- `--save-mask`: Save segmentation mask
- `--save-viz`: Save visualization image
- `--batch-size, -b INTEGER`: Batch size for multiple images [default: 1]

**Examples:**

```bash
# Process single image
earsegmentationai process-image image.jpg --save-viz

# Process directory with GPU
earsegmentationai process-image ./images --device cuda:0 --save-mask --save-viz

# Custom threshold
earsegmentationai process-image image.jpg --threshold 0.7 --output results/
```

### process-camera

Real-time camera/webcam processing.

```bash
earsegmentationai process-camera [OPTIONS]
```

**Options:**
- `--device-id, -i INTEGER`: Camera device ID [default: 0]
- `--device, -d TEXT`: Processing device [default: cpu]
- `--threshold, -t FLOAT`: Binary threshold [default: 0.5]
- `--save-video PATH`: Save output video
- `--fps INTEGER`: Output video FPS [default: 30]
- `--skip-frames INTEGER`: Frames to skip [default: 0]
- `--display/--no-display`: Show preview [default: display]

**Examples:**

```bash
# Basic webcam processing
earsegmentationai process-camera

# Save video with GPU processing
earsegmentationai process-camera --device cuda:0 --save-video output.mp4

# Process alternate frames for performance
earsegmentationai process-camera --skip-frames 1
```

### process-video

Process video files.

```bash
earsegmentationai process-video [OPTIONS] PATH
```

**Arguments:**
- `PATH`: Path to video file

**Options:**
- `--output, -o PATH`: Output video path
- `--device, -d TEXT`: Processing device [default: cpu]
- `--threshold, -t FLOAT`: Binary threshold [default: 0.5]
- `--skip-frames INTEGER`: Frames to skip [default: 0]
- `--display/--no-display`: Show preview [default: no-display]
- `--save-masks PATH`: Directory to save frame masks

**Examples:**

```bash
# Basic video processing
earsegmentationai process-video input.mp4 -o output.mp4

# Process with preview
earsegmentationai process-video input.mp4 --display

# Save individual masks
earsegmentationai process-video input.mp4 --save-masks masks/
```

### benchmark

Run performance benchmarks.

```bash
earsegmentationai benchmark [OPTIONS] PATH
```

**Arguments:**
- `PATH`: Path to test image

**Options:**
- `--device, -d TEXT`: Processing device [default: cpu]
- `--iterations, -n INTEGER`: Number of iterations [default: 100]
- `--warmup INTEGER`: Warmup iterations [default: 10]

**Examples:**

```bash
# CPU benchmark
earsegmentationai benchmark test.jpg

# GPU benchmark with more iterations
earsegmentationai benchmark test.jpg --device cuda:0 --iterations 1000
```

### download-model

Download the ear segmentation model.

```bash
earsegmentationai download-model [OPTIONS]
```

**Options:**
- `--force`: Force re-download even if exists
- `--model-dir PATH`: Custom model directory

**Examples:**

```bash
# Download model
earsegmentationai download-model

# Force re-download
earsegmentationai download-model --force
```

## Global Options

These options work with all commands:

- `--help`: Show help message
- `--quiet, -q`: Suppress info messages
- `--verbose, -v`: Show debug messages

## Configuration

Set defaults using environment variables:

```bash
export EARSEGMENTATIONAI_DEVICE="cuda:0"
export EARSEGMENTATIONAI_THRESHOLD="0.6"
export EARSEGMENTATIONAI_MODEL_DIR="/custom/models/"
```

Or using a config file at `~/.earsegmentationai/config.yaml`:

```yaml
processing:
  device: cuda:0
  threshold: 0.6
  batch_size: 4

video:
  fps: 30
  skip_frames: 1

paths:
  model_dir: /custom/models/
  cache_dir: /custom/cache/
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: File not found
- `4`: Model download error
- `5`: Processing error

## Keyboard Shortcuts

During camera/video preview:
- `q`: Quit
- `s`: Save screenshot
- `r`: Start/stop recording
- `SPACE`: Pause/resume
- `+/-`: Adjust threshold
