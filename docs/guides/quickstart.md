# Quick Start Guide

## Basic Usage

### Command Line Interface

#### Process a single image
```bash
earsegmentationai process-image path/to/image.jpg --save-viz
```

#### Process a directory of images
```bash
earsegmentationai process-image path/to/directory --save-viz --save-mask
```

#### Real-time webcam processing
```bash
earsegmentationai process-camera --device-id 0 --save-video output.mp4
```

### Python API

#### Basic example
```python
from earsegmentationai import ImageProcessor

# Initialize processor
processor = ImageProcessor()

# Process single image
result = processor.process("path/to/image.jpg")
print(f"Number of ears detected: {result.num_ears}")
```

#### Process with visualization
```python
# Process with visualization
result = processor.process(
    "path/to/image.jpg",
    return_visualization=True,
    save_results=True,
    output_dir="output"
)

# Access results
if result.success:
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Processing time: {result.processing_time:.3f}s")
```

#### Batch processing
```python
# Process multiple images
results = processor.process([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
])

for idx, result in enumerate(results.results):
    print(f"Image {idx}: {result.num_ears} ears detected")
```

## Common Use Cases

### 1. Save segmentation masks
```bash
earsegmentationai process-image image.jpg --save-mask --output masks/
```

### 2. Process with custom threshold
```bash
earsegmentationai process-image image.jpg --threshold 0.7
```

### 3. Use GPU acceleration
```bash
earsegmentationai process-image image.jpg --device cuda:0
```

### 4. Process video file
```python
from earsegmentationai import VideoProcessor

processor = VideoProcessor()
processor.process("input_video.mp4", output_path="output_video.mp4")
```

## Output Format

### CLI Output
```
Processing: image.jpg
✓ Ear detected!
Area: 1.55% of image
Bounding box: x=54, y=144, w=76, h=65
Results saved to: output/
```

### API Result Object
```python
result.success          # bool: Processing successful
result.num_ears         # int: Number of ears detected
result.mask            # numpy array: Segmentation mask
result.confidence      # float: Detection confidence
result.processing_time # float: Time in seconds
result.visualization   # numpy array: Visualization image (optional)
```

## Next Steps

- See [Advanced Usage](advanced.md) for more features
- Check [API Reference](../api/) for detailed documentation
- View [Examples](../../examples/) for more code samples