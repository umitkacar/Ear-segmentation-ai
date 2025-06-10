"""Unit tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from earsegmentationai.core.config import (
    Config,
    ModelConfig,
    ProcessingConfig,
    VideoConfig,
    get_config,
    load_config,
    set_config,
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.name == "earsegmentation_model_v1_46.pth"
        assert config.architecture == "Unet"
        assert config.encoder_name == "resnet18"
        assert config.classes == 1
        assert config.activation == "sigmoid"
    
    def test_architecture_validation(self):
        """Test architecture validation."""
        # Valid architecture
        config = ModelConfig(architecture="FPN")
        assert config.architecture == "FPN"
        
        # Invalid architecture
        with pytest.raises(ValueError, match="Architecture must be one of"):
            ModelConfig(architecture="InvalidArch")
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            name="custom_model.pth",
            architecture="DeepLabV3",
            encoder_name="resnet50",
            classes=2
        )
        assert config.name == "custom_model.pth"
        assert config.architecture == "DeepLabV3"
        assert config.encoder_name == "resnet50"
        assert config.classes == 2


class TestProcessingConfig:
    """Test ProcessingConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        assert config.input_size == (480, 320)
        assert config.batch_size == 1
        assert config.num_workers == 4
    
    def test_device_validation(self):
        """Test device validation."""
        # CPU device
        config = ProcessingConfig(device="cpu")
        assert config.device == "cpu"
        
        # CUDA device when not available
        import torch
        if not torch.cuda.is_available():
            config = ProcessingConfig(device="cuda:0")
            assert config.device == "cpu"  # Should fallback to CPU
    
    def test_input_size_validation(self):
        """Test input size validation."""
        # Valid input size
        config = ProcessingConfig(input_size=[640, 480])
        assert config.input_size == (640, 480)
        
        # Invalid input size
        with pytest.raises(ValueError, match="Input size must be a tuple"):
            ProcessingConfig(input_size=[640])


class TestVideoConfig:
    """Test VideoConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VideoConfig()
        assert config.fps == 30
        assert config.codec == "XVID"
        assert config.output_format == "avi"
        assert config.frame_size == (640, 480)
    
    def test_fps_validation(self):
        """Test FPS validation."""
        # Valid FPS
        config = VideoConfig(fps=60)
        assert config.fps == 60
        
        # Invalid FPS
        with pytest.raises(ValueError, match="FPS must be between"):
            VideoConfig(fps=0)
        
        with pytest.raises(ValueError, match="FPS must be between"):
            VideoConfig(fps=150)


class TestConfig:
    """Test main Config class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.video, VideoConfig)
        assert config.log_level == "INFO"
    
    def test_directory_creation(self, temp_dir):
        """Test automatic directory creation."""
        cache_dir = temp_dir / "test_cache"
        log_dir = temp_dir / "test_logs"
        
        config = Config(cache_dir=cache_dir, log_dir=log_dir)
        
        assert cache_dir.exists()
        assert log_dir.exists()
    
    def test_model_path_property(self):
        """Test model_path property."""
        config = Config()
        expected_path = config.cache_dir / "models" / config.model.name
        assert config.model_path == expected_path
    
    def test_from_yaml(self, temp_dir):
        """Test loading configuration from YAML."""
        # Create test YAML file
        yaml_path = temp_dir / "test_config.yaml"
        yaml_data = {
            "model": {
                "name": "test_model.pth",
                "architecture": "FPN"
            },
            "processing": {
                "batch_size": 4,
                "device": "cpu"
            },
            "log_level": "DEBUG"
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
        
        # Load config
        config = Config.from_yaml(yaml_path)
        
        assert config.model.name == "test_model.pth"
        assert config.model.architecture == "FPN"
        assert config.processing.batch_size == 4
        assert config.log_level == "DEBUG"
    
    def test_to_yaml(self, temp_dir):
        """Test saving configuration to YAML."""
        config = Config(
            model=ModelConfig(name="save_test.pth"),
            processing=ProcessingConfig(batch_size=8)
        )
        
        yaml_path = temp_dir / "saved_config.yaml"
        config.to_yaml(yaml_path)
        
        assert yaml_path.exists()
        
        # Load and verify
        with open(yaml_path, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["model"]["name"] == "save_test.pth"
        assert loaded_data["processing"]["batch_size"] == 8
    
    def test_yaml_file_not_found(self):
        """Test error when YAML file not found."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("non_existent_file.yaml")


class TestGlobalConfig:
    """Test global configuration functions."""
    
    def test_get_set_config(self):
        """Test get/set global config."""
        # Get default config
        config1 = get_config()
        assert isinstance(config1, Config)
        
        # Set new config
        new_config = Config(log_level="DEBUG")
        set_config(new_config)
        
        # Verify it was set
        config2 = get_config()
        assert config2.log_level == "DEBUG"
        assert config2 is new_config
    
    def test_load_config(self, temp_dir):
        """Test load_config function."""
        # Create test YAML file
        yaml_path = temp_dir / "load_test.yaml"
        yaml_data = {"log_level": "WARNING"}
        
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
        
        # Load and set config
        config = load_config(yaml_path)
        
        assert config.log_level == "WARNING"
        assert get_config().log_level == "WARNING"