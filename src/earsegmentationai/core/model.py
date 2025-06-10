"""Model management for Ear Segmentation AI."""

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Union

import requests
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from tqdm import tqdm

from earsegmentationai.core.config import get_config
from earsegmentationai.utils.exceptions import (
    DeviceError,
    InvalidModelError,
    ModelLoadError,
    ModelNotFoundError,
)
from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manages model loading, caching, and lifecycle.
    
    This class implements a singleton pattern to ensure only one model
    instance is loaded at a time, saving memory and improving performance.
    """
    
    _instance: Optional["ModelManager"] = None
    _model: Optional[nn.Module] = None
    _device: Optional[torch.device] = None
    
    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model manager."""
        if not hasattr(self, "_initialized"):
            self.config = get_config()
            self._initialized = True
            logger.info("ModelManager initialized")
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        if self._device is None:
            self._device = self._get_device(self.config.processing.device)
        return self._device
    
    @property
    def model(self) -> nn.Module:
        """Get or load model."""
        if self._model is None:
            self._model = self.load_model()
        return self._model
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get torch device from string.
        
        Args:
            device_str: Device string (e.g., "cpu", "cuda:0")
            
        Returns:
            torch.device instance
            
        Raises:
            DeviceError: If device is not available
        """
        try:
            device = torch.device(device_str)
            
            if device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = torch.device("cpu")
            elif device.type == "cuda":
                # Check if specific GPU is available
                if device.index is not None and device.index >= torch.cuda.device_count():
                    raise DeviceError(f"CUDA device {device.index} not available")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
            
            return device
            
        except Exception as e:
            raise DeviceError(f"Invalid device string '{device_str}': {e}")
    
    def _download_model(self, url: str, destination: Path, chunk_size: int = 8192) -> None:
        """Download model from URL.
        
        Args:
            url: Model URL
            destination: Destination file path
            chunk_size: Download chunk size
            
        Raises:
            ModelLoadError: If download fails
        """
        try:
            logger.info(f"Downloading model from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            
            with open(destination, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading model") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Model downloaded successfully to {destination}")
            
        except Exception as e:
            if destination.exists():
                destination.unlink()
            raise ModelLoadError(f"Failed to download model: {e}")
    
    def _verify_model(self, path: Path, expected_hash: Optional[str] = None) -> bool:
        """Verify model integrity.
        
        Args:
            path: Model file path
            expected_hash: Expected SHA256 hash (optional)
            
        Returns:
            True if verification passes
        """
        if not path.exists():
            return False
        
        if expected_hash:
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            
            actual_hash = sha256.hexdigest()
            if actual_hash != expected_hash:
                logger.error(f"Model hash mismatch: {actual_hash} != {expected_hash}")
                return False
        
        return True
    
    def _create_model(self) -> nn.Module:
        """Create model architecture.
        
        Returns:
            Model instance
            
        Raises:
            InvalidModelError: If model architecture is invalid
        """
        try:
            model_config = self.config.model
            
            # Get model class from segmentation_models_pytorch
            model_class = getattr(smp, model_config.architecture, None)
            if model_class is None:
                raise InvalidModelError(f"Unknown architecture: {model_config.architecture}")
            
            # Create model
            model = model_class(
                encoder_name=model_config.encoder_name,
                encoder_weights=model_config.encoder_weights,
                classes=model_config.classes,
                activation=model_config.activation,
            )
            
            logger.info(f"Created {model_config.architecture} model with {model_config.encoder_name} encoder")
            return model
            
        except Exception as e:
            raise InvalidModelError(f"Failed to create model: {e}")
    
    def load_model(self, force_download: bool = False) -> nn.Module:
        """Load model, downloading if necessary.
        
        Args:
            force_download: Force re-download even if model exists
            
        Returns:
            Loaded model
            
        Raises:
            ModelLoadError: If model cannot be loaded
        """
        try:
            model_path = self.config.model_path
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download if necessary
            if force_download or not model_path.exists():
                self._download_model(self.config.model.url, model_path)
            
            if not model_path.exists():
                raise ModelNotFoundError(f"Model not found at {model_path}")
            
            # Create model architecture
            model = self._create_model()
            
            # Load weights
            logger.info(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different formats
            if isinstance(checkpoint, dict):
                # New format with state dict
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                # Old format - direct model object
                logger.warning("Loading legacy model format")
                model = checkpoint
            
            model = model.to(self.device)
            model.eval()
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def reload_model(self) -> nn.Module:
        """Force reload model.
        
        Returns:
            Reloaded model
        """
        logger.info("Reloading model")
        self._model = None
        return self.model
    
    def set_device(self, device: Union[str, torch.device]) -> None:
        """Change device and move model.
        
        Args:
            device: New device
        """
        if isinstance(device, str):
            device = self._get_device(device)
        
        if self._device != device:
            logger.info(f"Moving model from {self._device} to {device}")
            self._device = device
            if self._model is not None:
                self._model = self._model.to(device)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "architecture": self.config.model.architecture,
            "encoder": self.config.model.encoder_name,
            "device": str(self.device),
            "loaded": self._model is not None,
        }
        
        if self._model is not None:
            info.update({
                "parameters": sum(p.numel() for p in self._model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self._model.parameters() if p.requires_grad),
                "memory_mb": sum(p.element_size() * p.nelement() for p in self._model.parameters()) / 1024 / 1024,
            })
        
        return info
    
    def clear_cache(self) -> None:
        """Clear CUDA cache if using GPU."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")