# Migration Guide: v1.x to v2.0

This guide helps you migrate from Ear Segmentation AI v1.x to v2.0.

## 🎯 Overview

Version 2.0 is a complete refactor with a modern architecture, but maintains full backward compatibility. Your existing v1.x code will continue to work with deprecation warnings.

## 🔄 Quick Migration

### Python API

#### Image Processing

**v1.x (Old Way)**
```python
from earsegmentationai.ear_models import EarModel

# Initialize
model = EarModel()
model.download_models()
model.load_model(device="cuda:0")

# Process image
mask_original, mask_resized = model.predict("image.jpg")
```

**v2.0 (New Way)**
```python
from earsegmentationai import ImageProcessor

# Initialize (auto-downloads model)
processor = ImageProcessor(device="cuda:0")

# Process image
result = processor.process("image.jpg")
mask = result.mask
has_ear = result.has_ear
ear_percentage = result.ear_percentage
```

#### Video/Camera Processing

**v1.x (Old Way)**
```python
from earsegmentationai.camera_mode import process_camera_mode

process_camera_mode(
    deviceId=0,
    device="cuda:0",
    record=True
)
```

**v2.0 (New Way)**
```python
from earsegmentationai import VideoProcessor

processor = VideoProcessor(device="cuda:0")
stats = processor.process(
    0,  # camera ID
    output_path="output.avi",
    display=True
)
```

### CLI Commands

**v1.x Commands**
```bash
python -m earsegmentationai.main picture-capture --folderpath image.jpg --device cuda:0
python -m earsegmentationai.main webcam-capture --deviceid 0 --device cuda:0
python -m earsegmentationai.main video-capture --filepath video.mp4
```

**v2.0 Commands**
```bash
earsegmentationai process-image image.jpg --device cuda:0
earsegmentationai webcam --device-id 0 --device cuda:0
earsegmentationai process-video video.mp4 --device cuda:0
```

## 📊 Feature Comparison

| Feature | v1.x | v2.0 |
|---------|------|------|
| Single image processing | ✅ | ✅ Enhanced |
| Batch processing | ❌ | ✅ |
| Video file processing | ✅ | ✅ Enhanced |
| Webcam processing | ✅ | ✅ Enhanced |
| URL image processing | ❌ | ✅ |
| Progress bars | ❌ | ✅ |
| Temporal smoothing | ❌ | ✅ |
| Configuration files | ❌ | ✅ |
| Type hints | ❌ | ✅ |
| Comprehensive tests | ❌ | ✅ |

## 🔧 New Features in v2.0

### 1. Batch Processing
```python
processor = ImageProcessor()
results = processor.process([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
])
print(f"Detection rate: {results.detection_rate}%")
```

### 2. Rich Processing Results
```python
result = processor.process("image.jpg")
print(f"Has ear: {result.has_ear}")
print(f"Ear area: {result.ear_percentage}%")
print(f"Bounding box: {result.get_bounding_box()}")
print(f"Center: {result.get_center()}")
```

### 3. Configuration Management
```python
from earsegmentationai import Config

config = Config(
    model={"architecture": "FPN"},
    processing={"batch_size": 8}
)
processor = ImageProcessor(config=config)
```

### 4. Performance Benchmarking
```bash
earsegmentationai benchmark image.jpg --iterations 100
```

## ⚠️ Deprecation Warnings

When using v1.x APIs, you'll see deprecation warnings:
```
DeprecationWarning: EarModel is deprecated. Use ImageProcessor or VideoProcessor instead.
```

To suppress these warnings temporarily:
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

## 🚀 Step-by-Step Migration

### Step 1: Update Imports
```python
# Old
from earsegmentationai.ear_models import EarModel
from earsegmentationai.camera_mode import process_camera_mode

# New
from earsegmentationai import ImageProcessor, VideoProcessor
```

### Step 2: Update Initialization
```python
# Old
model = EarModel()
model.download_models()
model.load_model("cuda:0")

# New
processor = ImageProcessor(device="cuda:0")
# Model downloads automatically on first use
```

### Step 3: Update Processing Code
```python
# Old
mask_orig, mask_resized = model.predict(image)

# New
result = processor.process(image)
mask = result.mask
```

### Step 4: Update CLI Scripts
Replace old CLI commands with new ones (see CLI Commands section above).

## 🔍 Common Issues

### Issue: Import Error
```python
ImportError: cannot import name 'EarModel' from 'earsegmentationai.ear_models'
```
**Solution**: The old modules are available through the compatibility layer:
```python
from earsegmentationai import EarModel  # Works with v2.0
```

### Issue: Different Output Format
**v1.x** returned tuple: `(mask_original, mask_resized)`
**v2.0** returns `ProcessingResult` object with properties

**Solution**: Access mask directly:
```python
# If you need tuple format for compatibility
result = processor.process(image)
mask_tuple = (result.mask, result.mask)  # Both are same in v2.0
```

### Issue: Missing Parameters
Some v1.x parameters like `foler_path` (typo) are deprecated.

**Solution**: These parameters are ignored in v2.0 with warnings.

## 💡 Best Practices

1. **Gradual Migration**: Your v1.x code will continue to work. Migrate gradually.

2. **Use New Features**: Take advantage of batch processing and rich results.

3. **Update Tests**: Write tests using the new API.

4. **Configuration Files**: Use YAML configuration for complex setups.

## 📚 Resources

- [API Documentation](docs/api/reference.md)
- [Examples](examples/)
- [GitHub Issues](https://github.com/umitkacar/Ear-segmentation-ai/issues)

## 🆘 Need Help?

If you encounter issues during migration:

1. Check this guide
2. Review the [CHANGELOG](CHANGELOG.md)
3. Look at [example scripts](examples/)
4. Open an issue on GitHub

Remember: v2.0 is fully backward compatible, so there's no rush to migrate everything at once!