# 🦻 Ear Segmentation AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/earsegmentationai)](https://pypi.org/project/earsegmentationai/)
[![Python](https://img.shields.io/pypi/pyversions/earsegmentationai)](https://pypi.org/project/earsegmentationai/)
[![Downloads](https://img.shields.io/pypi/dm/earsegmentationai)](https://pypi.org/project/earsegmentationai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

A state-of-the-art ear segmentation library powered by deep learning. Detect and segment human ears in images and video streams with high accuracy and real-time performance.

<p align="center">
  <img src="assets/images/demo.gif" alt="Ear Segmentation Demo" width="600">
</p>

## ✨ Features

- 🚀 **High Performance**: Optimized for both CPU and GPU processing
- 🎯 **Accurate Detection**: State-of-the-art U-Net architecture with ResNet18 encoder
- 📷 **Multiple Input Sources**: Images, videos, webcam, and URLs
- 🔄 **Real-time Processing**: Smooth webcam segmentation with temporal smoothing
- 📊 **Batch Processing**: Efficient processing of multiple images
- 🛠️ **Easy to Use**: Simple Python API and CLI interface
- 🎨 **Visualization Tools**: Built-in mask overlay and heatmap visualization
- 📦 **Lightweight**: Minimal dependencies, easy to install

## 🚀 Quick Start

### Installation

```bash
# Using pip
pip install earsegmentationai

# Using poetry (recommended)
poetry add earsegmentationai
```

For detailed installation instructions, see [Installation Guide](docs/guides/installation.md).

### Basic Usage

#### Python API

```python
from earsegmentationai import ImageProcessor

# Initialize processor
processor = ImageProcessor(device="cpu")  # or "cuda:0" for GPU

# Process single image
result = processor.process("path/to/image.jpg")
print(f"Ear detected: {result.has_ear}")
print(f"Ear area: {result.ear_percentage:.2f}% of image")

# Process with visualization
result = processor.process(
    "path/to/image.jpg",
    return_visualization=True
)
```

#### Command Line

```bash
# Process single image
earsegmentationai process-image path/to/image.jpg --save-viz

# Process directory
earsegmentationai process-image path/to/images/ -o output/

# Real-time webcam
earsegmentationai webcam --device cuda:0

# Process video
earsegmentationai process-video path/to/video.mp4 -o output.avi
```

## 📚 Documentation

- [Installation Guide](docs/guides/installation.md)
- [Quick Start Guide](docs/guides/quickstart.md) 
- [Architecture Overview](docs/development/architecture.md)
- [API Reference](docs/api/)
- [Contributing Guide](docs/development/CONTRIBUTING.md)
- [Migration Guide](docs/migration/MIGRATION.md)

## 📚 Advanced Usage

### Batch Processing

```python
from earsegmentationai import ImageProcessor

processor = ImageProcessor(device="cuda:0")

# Process multiple images
results = processor.process([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
])

print(f"Detection rate: {results.detection_rate:.1f}%")
print(f"Average ear area: {results.average_ear_area:.0f} pixels")
```

### Video Processing

```python
from earsegmentationai import VideoProcessor

processor = VideoProcessor(
    device="cuda:0",
    skip_frames=2,      # Process every 3rd frame
    smooth_masks=True   # Temporal smoothing
)

# Process video file
stats = processor.process(
    "video.mp4",
    output_path="output.mp4",
    display=True
)

print(f"FPS: {stats['average_fps']:.1f}")
print(f"Detection rate: {stats['detection_rate']:.1f}%")
```

### Custom Configuration

```python
from earsegmentationai import ImageProcessor, Config

# Create custom configuration
config = Config(
    model={"architecture": "FPN", "encoder_name": "resnet50"},
    processing={"input_size": (640, 480), "batch_size": 8}
)

processor = ImageProcessor(config=config, threshold=0.7)
```

## 🔧 Configuration

### Model Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `architecture` | `"Unet"` | Model architecture (Unet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus) |
| `encoder_name` | `"resnet18"` | Encoder backbone |
| `input_size` | `(480, 320)` | Input image size (width, height) |
| `threshold` | `0.5` | Binary mask threshold |

### Processing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"cpu"` | Processing device (cpu, cuda:0) |
| `batch_size` | `1` | Batch size for processing |
| `skip_frames` | `0` | Frame skipping for video (0 = process all) |
| `smooth_masks` | `True` | Enable temporal smoothing for video |

## 🏗️ Architecture

The library uses a modular architecture with clear separation of concerns:

```
earsegmentationai/
├── core/           # Core model and prediction logic
├── preprocessing/  # Image preprocessing and validation
├── postprocessing/ # Visualization and export utilities
├── api/           # High-level Python API
├── cli/           # Command-line interface
└── utils/         # Logging, exceptions, and helpers
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test suite
poetry run pytest tests/unit/test_transforms.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/CONTRIBUTING.md) for details.

```bash
# Setup development environment
make install-dev

# Run linting and formatting
make format
make lint

# Run pre-commit hooks
make pre-commit
```

## 📈 Performance

| Device | Image Size | FPS | Memory |
|--------|------------|-----|---------|
| CPU (i7-9700K) | 480×320 | 15 | 200 MB |
| GPU (RTX 3080) | 480×320 | 120 | 400 MB |
| GPU (RTX 3080) | 1920×1080 | 45 | 800 MB |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- Inspired by state-of-the-art segmentation research
- Thanks to all contributors and the open-source community

## 📞 Support

- 📧 Email: umitkacar.phd@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/umitkacar/Ear-segmentation-ai/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/umitkacar/Ear-segmentation-ai/discussions)

---

<p align="center">Made with ❤️ by the Ear Segmentation AI Team</p>
