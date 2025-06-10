# Configuration API

## Config Classes

### Config

Main configuration class for the entire application.

```python
from earsegmentationai.core.config import Config
```

**Attributes:**
- `model` (ModelConfig): Model configuration
- `processing` (ProcessingConfig): Processing configuration
- `video` (VideoConfig): Video configuration
- `paths` (PathsConfig): Path configuration
- `logging` (LoggingConfig): Logging configuration

**Methods:**

```python
# Load from YAML file
config = Config.from_yaml("config.yaml")

# Save to YAML file
config.to_yaml("config.yaml")

# Get model path
model_path = config.model_path

# Get default config
config = Config()
```

### ModelConfig

Model-specific configuration.

```python
from earsegmentationai.core.config import ModelConfig
```

**Attributes:**
- `architecture` (str): Model architecture [default: "Unet"]
- `encoder_name` (str): Encoder name [default: "resnet18"]
- `encoder_weights` (str): Encoder weights [default: "imagenet"]
- `in_channels` (int): Input channels [default: 3]
- `classes` (int): Output classes [default: 1]
- `download_url` (str): Model download URL
- `filename` (str): Model filename

### ProcessingConfig

Processing configuration.

```python
from earsegmentationai.core.config import ProcessingConfig
```

**Attributes:**
- `device` (str): Processing device [default: "cpu"]
- `input_size` (Tuple[int, int]): Model input size [default: (480, 320)]
- `batch_size` (int): Batch size [default: 1]
- `threshold` (float): Binary threshold [default: 0.5]
- `normalize_mean` (Tuple[float, ...]): Normalization mean
- `normalize_std` (Tuple[float, ...]): Normalization std

### VideoConfig

Video processing configuration.

```python
from earsegmentationai.core.config import VideoConfig
```

**Attributes:**
- `fps` (int): Output FPS [default: 30]
- `codec` (str): Video codec [default: "mp4v"]
- `skip_frames` (int): Frames to skip [default: 0]
- `buffer_size` (int): Frame buffer size [default: 10]

### PathsConfig

Path configuration.

```python
from earsegmentationai.core.config import PathsConfig
```

**Attributes:**
- `base_dir` (Path): Base directory
- `model_dir` (Path): Model directory
- `cache_dir` (Path): Cache directory
- `output_dir` (Path): Default output directory

## Global Configuration

### get_config() / set_config()

Get or set global configuration.

```python
from earsegmentationai.core.config import get_config, set_config

# Get current config
config = get_config()

# Modify and set
config.processing.device = "cuda:0"
set_config(config)
```

### load_config()

Load configuration from file or environment.

```python
from earsegmentationai.core.config import load_config

# Load from default locations
config = load_config()

# Load from specific file
config = load_config("custom_config.yaml")
```

## Configuration Files

### YAML Format

```yaml
model:
  architecture: Unet
  encoder_name: resnet18
  encoder_weights: imagenet
  in_channels: 3
  classes: 1

processing:
  device: cuda:0
  input_size: [480, 320]
  batch_size: 4
  threshold: 0.5
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]

video:
  fps: 30
  codec: mp4v
  skip_frames: 1
  buffer_size: 10

paths:
  model_dir: ~/.cache/earsegmentationai/models
  cache_dir: ~/.cache/earsegmentationai
  output_dir: ./output

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Environment Variables

Configuration can be overridden using environment variables:

```bash
# Model config
export EARSEGMENTATIONAI_MODEL_ARCHITECTURE="Unet"
export EARSEGMENTATIONAI_MODEL_ENCODER="resnet34"

# Processing config
export EARSEGMENTATIONAI_DEVICE="cuda:0"
export EARSEGMENTATIONAI_THRESHOLD="0.6"
export EARSEGMENTATIONAI_BATCH_SIZE="8"

# Path config
export EARSEGMENTATIONAI_MODEL_DIR="/custom/models"
export EARSEGMENTATIONAI_CACHE_DIR="/custom/cache"

# Logging
export EARSEGMENTATIONAI_LOG_LEVEL="DEBUG"
```

## Examples

### Basic Usage

```python
from earsegmentationai.core.config import Config

# Create default config
config = Config()

# Modify settings
config.processing.device = "cuda:0"
config.processing.threshold = 0.7

# Save to file
config.to_yaml("my_config.yaml")
```

### Custom Configuration

```python
from earsegmentationai.core.config import (
    Config, ModelConfig, ProcessingConfig
)

# Create custom config
config = Config(
    model=ModelConfig(
        encoder_name="resnet34",
        encoder_weights="imagenet"
    ),
    processing=ProcessingConfig(
        device="cuda:0",
        batch_size=8,
        threshold=0.6
    )
)
```

### Loading Configuration

```python
# From file
config = Config.from_yaml("config.yaml")

# From environment
from earsegmentationai.core.config import load_config
config = load_config()

# Merge multiple sources
base_config = Config()
file_config = Config.from_yaml("config.yaml")
config = base_config.merge(file_config)
```

### Validation

```python
from earsegmentationai.core.config import Config
from pydantic import ValidationError

try:
    config = Config(
        processing={"device": "invalid-device"}
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
```
