# Base Processing API

## BaseProcessor

The `BaseProcessor` provides shared functionality for all processing interfaces. It handles configuration loading, model initialization and common helper methods such as threshold and device management.

### Responsibilities
- manage configuration and override options
- instantiate the `ModelManager`, pre-processing transforms and prediction utilities
- provide reusable helpers (`set_device`, `set_threshold`, `get_info`, `warmup`, `clear_cache`)
- define the abstract `process()` method implemented by subclasses

### Usage
```python
from earsegmentationai.api.base import BaseProcessor

class CustomProcessor(BaseProcessor):
    def process(self, path: str):
        image = cv2.imread(path)
        mask = self.predictor.predict(image)
        return ProcessingResult(image, mask)

processor = CustomProcessor()
result = processor.process("image.jpg")
print(result.has_ear)
```

### Derived Classes
- **ImageProcessor** – handles static image files and arrays
- **VideoProcessor** – operates on video streams and frame sequences

These classes implement the `process()` method to support their respective data sources.

### Extension Points
Subclasses can customise behaviour by:
- overriding `process()` to accept new input types
- replacing the transformation pipeline (`self.transform`)
- injecting a custom `ModelManager` for advanced model control

## Processing Results
`ProcessingResult` encapsulates the original image, predicted mask and optional metadata. It offers convenience methods such as `has_ear`, `ear_area`, `ear_percentage`, `get_bounding_box`, and `get_center` to inspect predictions.

