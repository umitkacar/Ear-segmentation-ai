"""Configuration management for Ear Segmentation AI."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ModelConfig(BaseModel):
    """Model configuration."""
    model_config = ConfigDict(validate_assignment=True)
    
    name: str = Field(default="earsegmentation_model_v1_46.pth", description="Model filename")
    url: str = Field(
        default="https://github.com/umitkacar/Ear-segmentation-ai/releases/download/v1.0.0/earsegmentation_model_v1_46.pth",
        description="Model download URL"
    )
    architecture: str = Field(default="Unet", description="Model architecture")
    encoder_name: str = Field(default="resnet18", description="Encoder name")
    encoder_weights: str = Field(default="imagenet", description="Encoder weights")
    classes: int = Field(default=1, description="Number of output classes")
    activation: Optional[str] = Field(default="sigmoid", description="Output activation")
    
    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v):
        """Validate model architecture."""
        valid_architectures = ["Unet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus"]
        if v not in valid_architectures:
            raise ValueError(f"Architecture must be one of {valid_architectures}")
        return v


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    model_config = ConfigDict(validate_assignment=True)
    
    input_size: tuple = Field(default=(480, 320), description="Input image size (width, height)")
    batch_size: int = Field(default=1, description="Batch size for processing")
    device: str = Field(default="cuda:0", description="Processing device")
    num_workers: int = Field(default=4, description="Number of data loading workers")
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        """Validate device string."""
        import torch
        if v.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return v
    
    @field_validator("input_size")
    @classmethod
    def validate_input_size(cls, v):
        """Validate input size."""
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError("Input size must be a tuple of (width, height)")
        return tuple(v)


class VideoConfig(BaseModel):
    """Video processing configuration."""
    model_config = ConfigDict(validate_assignment=True)
    
    fps: int = Field(default=30, description="Frames per second")
    codec: str = Field(default="XVID", description="Video codec")
    output_format: str = Field(default="avi", description="Output video format")
    frame_size: tuple = Field(default=(640, 480), description="Output frame size")
    
    @field_validator("fps")
    @classmethod
    def validate_fps(cls, v):
        """Validate FPS value."""
        if v <= 0 or v > 120:
            raise ValueError("FPS must be between 1 and 120")
        return v


class Config(BaseModel):
    """Main configuration class."""
    model_config = ConfigDict(validate_assignment=True)
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    
    # Paths
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "earsegmentationai",
        description="Cache directory for models"
    )
    log_dir: Path = Field(
        default_factory=lambda: Path.home() / ".logs" / "earsegmentationai",
        description="Log directory"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    @field_validator("cache_dir", "log_dir")
    @classmethod
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict with Path objects as strings
        data = self.model_dump()
        # Convert Path objects to strings and tuples to lists for YAML serialization
        def convert_for_yaml(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_yaml(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_yaml(item) for item in obj]
            return obj
        
        data = convert_for_yaml(data)
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @property
    def model_path(self) -> Path:
        """Get full model path."""
        return self.cache_dir / "models" / self.model.name


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config


def load_config(path: Union[str, Path]) -> Config:
    """Load and set global configuration from file."""
    config = Config.from_yaml(path)
    set_config(config)
    return config