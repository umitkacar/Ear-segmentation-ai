# Type Definitions

## Core Types

### ProcessingResult

Result from single image processing.

```python
@dataclass
class ProcessingResult:
    success: bool
    mask: Optional[np.ndarray]
    probability_map: Optional[np.ndarray]
    visualization: Optional[np.ndarray]
    num_ears: int
    confidence: float
    processing_time: float
    ear_percentage: float
    bounding_box: Optional[Tuple[int, int, int, int]]
    metadata: Dict[str, Any]
```

### BatchProcessingResult

Result from batch image processing.

```python
@dataclass
class BatchProcessingResult:
    results: List[ProcessingResult]
    total_images: int
    successful_images: int
    failed_images: int
    total_time: float
    average_time: float
    detection_rate: float
    metadata: Dict[str, Any]
```

### VideoStats

Statistics from video processing.

```python
@dataclass
class VideoStats:
    total_frames: int
    frames_with_ears: int
    detection_rate: float
    average_fps: float
    total_time: float
    output_path: Optional[str]
    frame_dimensions: Tuple[int, int]
    metadata: Dict[str, Any]
```

### CameraStats

Statistics from camera processing.

```python
@dataclass 
class CameraStats:
    session_duration: float
    total_frames: int
    frames_with_ears: int
    detection_rate: float
    average_fps: float
    recording_path: Optional[str]
    metadata: Dict[str, Any]
```

## Validation Types

### ImageValidationResult

Result from image validation.

```python
@dataclass
class ImageValidationResult:
    is_valid: bool
    width: int
    height: int
    channels: int
    format: str
    error: Optional[str]
```

### VideoValidationResult

Result from video validation.

```python
@dataclass
class VideoValidationResult:
    is_valid: bool
    frame_count: int
    fps: float
    width: int
    height: int
    codec: str
    duration: float
    error: Optional[str]
```

## Transform Types

### TransformParams

Parameters for image transformations.

```python
@dataclass
class TransformParams:
    resize: Tuple[int, int]
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]
    pad_if_needed: bool
    pad_value: int
```

### AugmentationParams

Parameters for data augmentation.

```python
@dataclass
class AugmentationParams:
    horizontal_flip: bool
    vertical_flip: bool
    rotation_limit: int
    brightness_limit: float
    contrast_limit: float
    blur_limit: int
```

## Model Types

### ModelInfo

Information about loaded model.

```python
@dataclass
class ModelInfo:
    architecture: str
    encoder: str
    input_size: Tuple[int, int]
    parameters: int
    device: str
    memory_usage: float
    load_time: float
```

### InferenceMetrics

Metrics from model inference.

```python
@dataclass
class InferenceMetrics:
    preprocess_time: float
    inference_time: float
    postprocess_time: float
    total_time: float
    memory_peak: float
```

## Exception Types

### ProcessingError

Base exception for processing errors.

```python
class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass
```

### ModelError

Model-related errors.

```python
class ModelError(ProcessingError):
    """Model loading or inference error."""
    pass
```

### ValidationError

Input validation errors.

```python
class ValidationError(ProcessingError):
    """Input validation error."""
    pass
```

### ConfigurationError

Configuration errors.

```python
class ConfigurationError(ProcessingError):
    """Configuration error."""
    pass
```

## Enum Types

### DeviceType

Available device types.

```python
class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    CUDA0 = "cuda:0"
    CUDA1 = "cuda:1"
    MPS = "mps"
```

### OutputFormat

Output format options.

```python
class OutputFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    NUMPY = "numpy"
    TENSOR = "tensor"
```

### VisualizationType

Visualization types.

```python
class VisualizationType(str, Enum):
    OVERLAY = "overlay"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    SIDE_BY_SIDE = "side_by_side"
```

## Type Aliases

Common type aliases used throughout the codebase.

```python
# Image types
ImageArray = np.ndarray
ImagePath = Union[str, Path]
ImageInput = Union[ImagePath, ImageArray]

# Result types
MaskArray = np.ndarray
ProbabilityMap = np.ndarray
BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)

# Config types
DeviceStr = str
ThresholdFloat = float  # 0.0 to 1.0
BatchSizeInt = int  # >= 1

# Callback types
FrameCallback = Callable[[int, bool, float], None]
ProgressCallback = Callable[[float], None]
```

## Generic Types

### Result[T]

Generic result type for operations.

```python
@dataclass
class Result(Generic[T]):
    value: Optional[T]
    success: bool
    error: Optional[str]
    
    @property
    def is_ok(self) -> bool:
        return self.success
    
    def unwrap(self) -> T:
        if not self.success:
            raise ValueError(self.error)
        return self.value
```

### Pipeline[T, U]

Generic pipeline type.

```python
class Pipeline(Generic[T, U]):
    def __init__(self, steps: List[Callable[[T], T]]):
        self.steps = steps
    
    def process(self, input_data: T) -> U:
        result = input_data
        for step in self.steps:
            result = step(result)
        return result
```