"""Ear Segmentation AI - A PyTorch-based ear segmentation library.

This package provides tools for detecting and segmenting ears in images and video streams
using deep learning models.
"""

from earsegmentationai.__version__ import __version__
from earsegmentationai.api.image import ImageProcessor
from earsegmentationai.api.video import VideoProcessor
from earsegmentationai.core.model import ModelManager

# Legacy imports for backward compatibility
from earsegmentationai.compat import (
    EarModel,
    process_camera_mode,
    process_image_mode,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    MODEL_NAME,
    MODEL_URL,
)

__all__ = [
    "__version__",
    "ImageProcessor",
    "VideoProcessor",
    "ModelManager",
    # Legacy exports
    "EarModel",
    "process_camera_mode",
    "process_image_mode",
    "ENCODER_NAME",
    "ENCODER_WEIGHTS",
    "MODEL_NAME",
    "MODEL_URL",
]
