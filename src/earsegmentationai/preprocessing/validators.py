"""Input validation utilities."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from earsegmentationai.utils.exceptions import InvalidInputError, ValidationError
from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def validate_image_path(path: Union[str, Path]) -> Path:
    """Validate image file path.
    
    Args:
        path: Image file path
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise ValidationError(f"Image file not found: {path}")
    
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValidationError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}"
        )
    
    return path


def validate_video_path(path: Union[str, Path]) -> Path:
    """Validate video file path.
    
    Args:
        path: Video file path
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise ValidationError(f"Video file not found: {path}")
    
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValidationError(
            f"Unsupported video format: {path.suffix}. "
            f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}"
        )
    
    return path


def validate_directory(path: Union[str, Path], create: bool = False) -> Path:
    """Validate directory path.
    
    Args:
        path: Directory path
        create: Whether to create directory if it doesn't exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    path = Path(path)
    
    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        else:
            raise ValidationError(f"Directory not found: {path}")
    
    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")
    
    return path


def validate_image_array(image: np.ndarray) -> np.ndarray:
    """Validate image array.
    
    Args:
        image: Image array
        
    Returns:
        Validated image array
        
    Raises:
        InvalidInputError: If image is invalid
    """
    if not isinstance(image, np.ndarray):
        raise InvalidInputError(f"Image must be numpy array, got {type(image)}")
    
    if image.size == 0:
        raise InvalidInputError("Image is empty")
    
    if image.ndim not in [2, 3]:
        raise InvalidInputError(f"Image must be 2D or 3D array, got {image.ndim}D")
    
    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        raise InvalidInputError(
            f"Image must have 1, 3, or 4 channels, got {image.shape[2]}"
        )
    
    if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        logger.warning(f"Unusual image dtype: {image.dtype}")
    
    return image


def validate_mask_array(mask: np.ndarray, num_classes: int = 1) -> np.ndarray:
    """Validate mask array.
    
    Args:
        mask: Mask array
        num_classes: Number of classes
        
    Returns:
        Validated mask array
        
    Raises:
        InvalidInputError: If mask is invalid
    """
    if not isinstance(mask, np.ndarray):
        raise InvalidInputError(f"Mask must be numpy array, got {type(mask)}")
    
    if mask.size == 0:
        raise InvalidInputError("Mask is empty")
    
    if mask.ndim not in [2, 3]:
        raise InvalidInputError(f"Mask must be 2D or 3D array, got {mask.ndim}D")
    
    if mask.ndim == 3 and mask.shape[2] != num_classes:
        raise InvalidInputError(
            f"Mask must have {num_classes} channels, got {mask.shape[2]}"
        )
    
    # Check value range
    unique_values = np.unique(mask)
    if num_classes == 1:
        # Binary mask
        if not np.all(np.isin(unique_values, [0, 1])):
            logger.warning(f"Binary mask contains values other than 0 and 1: {unique_values}")
    else:
        # Multi-class mask
        max_value = unique_values.max()
        if max_value >= num_classes:
            raise InvalidInputError(
                f"Mask contains class {max_value} but only {num_classes} classes expected"
            )
    
    return mask


def validate_device_string(device: str) -> str:
    """Validate device string.
    
    Args:
        device: Device string (e.g., "cpu", "cuda:0")
        
    Returns:
        Validated device string
        
    Raises:
        ValidationError: If device string is invalid
    """
    import torch
    
    if device == "cpu":
        return device
    
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU instead")
            return "cpu"
        
        # Parse device index
        if device == "cuda":
            return device
        
        try:
            parts = device.split(":")
            if len(parts) == 2:
                device_idx = int(parts[1])
                if device_idx >= torch.cuda.device_count():
                    raise ValidationError(
                        f"CUDA device {device_idx} not available. "
                        f"Available devices: 0-{torch.cuda.device_count()-1}"
                    )
            else:
                raise ValidationError(f"Invalid CUDA device format: {device}")
        except ValueError:
            raise ValidationError(f"Invalid CUDA device format: {device}")
    else:
        raise ValidationError(
            f"Unknown device: {device}. Use 'cpu' or 'cuda:N'"
        )
    
    return device


def validate_batch_size(batch_size: int) -> int:
    """Validate batch size.
    
    Args:
        batch_size: Batch size
        
    Returns:
        Validated batch size
        
    Raises:
        ValidationError: If batch size is invalid
    """
    if not isinstance(batch_size, int):
        raise ValidationError(f"Batch size must be integer, got {type(batch_size)}")
    
    if batch_size <= 0:
        raise ValidationError(f"Batch size must be positive, got {batch_size}")
    
    if batch_size > 256:
        logger.warning(f"Large batch size: {batch_size}. This may cause memory issues.")
    
    return batch_size


def validate_camera_id(camera_id: Union[int, str]) -> Union[int, str]:
    """Validate camera ID.
    
    Args:
        camera_id: Camera ID (integer or string path)
        
    Returns:
        Validated camera ID
        
    Raises:
        ValidationError: If camera ID is invalid
    """
    if isinstance(camera_id, int):
        if camera_id < 0:
            raise ValidationError(f"Camera ID must be non-negative, got {camera_id}")
        
        # Try to open camera to validate
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            cap.release()
            raise ValidationError(f"Cannot open camera {camera_id}")
        cap.release()
        
    elif isinstance(camera_id, str):
        # Could be a device path like "/dev/video0" or URL
        if camera_id.startswith("/dev/"):
            if not Path(camera_id).exists():
                raise ValidationError(f"Camera device not found: {camera_id}")
        elif not (camera_id.startswith("http") or camera_id.startswith("rtsp")):
            raise ValidationError(f"Invalid camera path: {camera_id}")
    else:
        raise ValidationError(
            f"Camera ID must be integer or string, got {type(camera_id)}"
        )
    
    return camera_id


def get_image_files(directory: Union[str, Path]) -> List[Path]:
    """Get all image files in a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        List of image file paths
    """
    directory = validate_directory(directory)
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    # Sort by name
    image_files = sorted(set(image_files))
    
    if not image_files:
        logger.warning(f"No image files found in {directory}")
    
    return image_files