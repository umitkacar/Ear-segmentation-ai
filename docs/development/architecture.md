# System Architecture

## Overview

The Ear Segmentation AI system is built with a modular architecture that separates concerns and allows for easy extension and maintenance.

```
┌─────────────────────────────────────────────────────────────┐
│                          CLI Layer                          │
│                    (Typer + Rich CLI)                       │
├─────────────────────────────────────────────────────────────┤
│                          API Layer                          │
│              (ImageProcessor, VideoProcessor)               │
├─────────────────────────────────────────────────────────────┤
│                         Core Layer                          │
│           (ModelManager, EarPredictor, Config)              │
├─────────────────────────────────────────────────────────────┤
│                    Processing Pipeline                      │
│     (Preprocessing → Inference → Postprocessing)           │
├─────────────────────────────────────────────────────────────┤
│                      Infrastructure                         │
│           (Logging, Exceptions, Validators)                │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Management (`core/model.py`)
- **ModelManager**: Singleton pattern for model lifecycle
- Handles model downloading, loading, and caching
- Supports both CPU and GPU inference
- Thread-safe model access

### 2. Prediction Engine (`core/predictor.py`)
- **EarPredictor**: Core inference logic
- Preprocessing pipeline with Albumentations
- Batch processing support
- Post-processing with confidence thresholding

### 3. Configuration (`core/config.py`)
- **Config**: Pydantic-based configuration
- Environment variable support
- YAML serialization/deserialization
- Validation and type safety

### 4. API Layer (`api/`)
- **BaseProcessor**: Abstract base class
- **ImageProcessor**: Image processing API
- **VideoProcessor**: Video processing API
- Unified result types

### 5. CLI Interface (`cli/`)
- **app.py**: Main CLI application
- Command structure:
  - `version`: Show version info
  - `process-image`: Process images
  - `process-camera`: Real-time camera
  - `benchmark`: Performance testing

## Data Flow

### Image Processing Pipeline

```python
Input Image → Validation → Preprocessing → Model Inference → Post-processing → Results
     ↓             ↓              ↓               ↓                ↓            ↓
   Load      Check size    Resize/Norm    U-Net Forward    Threshold      Mask +
  Image      & format      to 480×320       Pass           Binary Mask   Metadata
```

### Video Processing Pipeline

```python
Video Input → Frame Extraction → Batch Processing → Frame Assembly → Output Video
      ↓              ↓                  ↓                 ↓              ↓
   Open       Read frames         Process each      Add overlays    Write with
  Stream      sequentially         with model       if requested      codec
```

## Model Architecture

### U-Net with ResNet18 Encoder
- **Encoder**: Pretrained ResNet18 backbone
- **Decoder**: Symmetric upsampling path
- **Skip Connections**: Feature concatenation
- **Output**: Single channel probability map

### Input/Output Specifications
- **Input**: RGB image (any size, resized internally)
- **Processing size**: 480×320 pixels
- **Output**: Binary mask (same size as input)

## Design Patterns

### 1. Singleton Pattern
Used in ModelManager to ensure single model instance:
```python
class ModelManager:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 2. Factory Pattern
Processor creation based on input type:
```python
def create_processor(input_type: str) -> BaseProcessor:
    if input_type == "image":
        return ImageProcessor()
    elif input_type == "video":
        return VideoProcessor()
```

### 3. Strategy Pattern
Different processing strategies for various input types while maintaining consistent interface.

## Extension Points

### 1. Adding New Models
- Implement model loading in ModelManager
- Update Config with new model parameters
- Maintain backward compatibility

### 2. Custom Preprocessing
- Add transforms to preprocessing pipeline
- Implement in `preprocessing/transforms.py`
- Register in predictor configuration

### 3. New Output Formats
- Extend ProcessingResult dataclass
- Add serialization methods
- Update API processors

## Performance Considerations

### 1. Memory Management
- Lazy model loading
- Batch processing for efficiency
- Automatic garbage collection

### 2. GPU Optimization
- CUDA memory management
- Mixed precision support (future)
- Batch size adaptation

### 3. Caching Strategy
- Model caching in ~/.cache
- Result caching (optional)
- Preprocessing cache

## Security Considerations

### 1. Input Validation
- File type verification
- Size limits
- Path traversal prevention

### 2. Model Integrity
- SHA256 checksum verification
- HTTPS download only
- Local cache validation

### 3. Output Sanitization
- Safe file naming
- Directory creation checks
- Permission validation