"""Base API classes for ear segmentation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from earsegmentationai.core.config import Config, get_config
from earsegmentationai.core.model import ModelManager
from earsegmentationai.core.predictor import EarPredictor
from earsegmentationai.postprocessing.visualization import MaskVisualizer
from earsegmentationai.preprocessing.transforms import ImageTransform
from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)


class BaseProcessor(ABC):
    """Base processor class for ear segmentation."""

    def __init__(
        self,
        config: Optional[Config] = None,
        model_manager: Optional[ModelManager] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        batch_size: int = 1,
    ):
        """Initialize base processor.

        Args:
            config: Configuration object. If None, uses default.
            model_manager: Model manager. If None, creates new one.
            device: Processing device. If None, uses config default.
            threshold: Binary threshold for mask generation.
            batch_size: Batch size for processing multiple images.
        """
        self.config = config or get_config()

        # Override device if specified
        if device:
            self.config.processing.device = device
        
        # Override batch_size if specified
        if batch_size != 1:
            self.config.processing.batch_size = batch_size

        # Initialize components
        self.model_manager = model_manager or ModelManager()
        self.transform = ImageTransform(
            input_size=self.config.processing.input_size
        )
        self.predictor = EarPredictor(
            model_manager=self.model_manager,
            transform=self.transform,
            threshold=threshold,
        )
        self.visualizer = MaskVisualizer()

        logger.info(
            f"{self.__class__.__name__} initialized with device={self.config.processing.device}"
        )

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process input data.

        This method should be implemented by subclasses.
        """
        pass

    def set_threshold(self, threshold: float) -> None:
        """Set binary threshold.

        Args:
            threshold: New threshold value (0-1)
        """
        self.predictor.set_threshold(threshold)

    def set_device(self, device: str) -> None:
        """Change processing device.

        Args:
            device: New device string
        """
        self.config.processing.device = device
        self.model_manager.set_device(device)

    def get_info(self) -> Dict[str, Any]:
        """Get processor information.

        Returns:
            Dictionary with processor info
        """
        return {
            "processor": self.__class__.__name__,
            "config": {
                "device": self.config.processing.device,
                "input_size": self.config.processing.input_size,
                "threshold": self.predictor.threshold,
            },
            "model": self.model_manager.get_model_info(),
        }

    def warmup(self) -> None:
        """Warmup the model."""
        self.predictor.warmup()

    def clear_cache(self) -> None:
        """Clear any caches."""
        self.model_manager.clear_cache()


class ProcessingResult:
    """Container for processing results."""

    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        probability_map: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize processing result.

        Args:
            image: Original image
            mask: Binary segmentation mask
            probability_map: Optional probability map
            metadata: Optional metadata dictionary
        """
        self.image = image
        self.mask = mask
        self.probability_map = probability_map
        self.metadata = metadata or {}

    @property
    def has_ear(self) -> bool:
        """Check if ear was detected."""
        return np.any(self.mask > 0)

    @property
    def ear_area(self) -> int:
        """Get ear area in pixels."""
        return np.sum(self.mask > 0)

    @property
    def ear_percentage(self) -> float:
        """Get ear area as percentage of image."""
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        return (self.ear_area / total_pixels) * 100

    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of ear region.

        Returns:
            Tuple of (x, y, width, height) or None if no ear
        """
        if not self.has_ear:
            return None

        # Find non-zero pixels
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)

        # Get bounds
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    def get_center(self) -> Optional[Tuple[int, int]]:
        """Get center of ear region.

        Returns:
            Tuple of (x, y) or None if no ear
        """
        bbox = self.get_bounding_box()
        if bbox is None:
            return None

        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation
        """
        result = {
            "has_ear": self.has_ear,
            "ear_area": self.ear_area,
            "ear_percentage": self.ear_percentage,
            "bounding_box": self.get_bounding_box(),
            "center": self.get_center(),
            "image_shape": self.image.shape,
            "metadata": self.metadata,
        }

        return result


class BatchProcessingResult:
    """Container for batch processing results."""

    def __init__(self, results: List[ProcessingResult]):
        """Initialize batch result.

        Args:
            results: List of individual results
        """
        self.results = results

    def __len__(self) -> int:
        """Get number of results."""
        return len(self.results)

    def __getitem__(self, index: int) -> ProcessingResult:
        """Get individual result by index."""
        return self.results[index]

    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)

    @property
    def detection_rate(self) -> float:
        """Get percentage of images with detected ears."""
        if not self.results:
            return 0.0

        detections = sum(1 for r in self.results if r.has_ear)
        return (detections / len(self.results)) * 100

    @property
    def average_ear_area(self) -> float:
        """Get average ear area across all detections."""
        ear_areas = [r.ear_area for r in self.results if r.has_ear]
        if not ear_areas:
            return 0.0

        return np.mean(ear_areas)

    def filter_by_detection(
        self, detected: bool = True
    ) -> List[ProcessingResult]:
        """Filter results by detection status.

        Args:
            detected: If True, return only detected. If False, only non-detected.

        Returns:
            Filtered results
        """
        return [r for r in self.results if r.has_ear == detected]

    def get_summary(self) -> Dict[str, Any]:
        """Get batch processing summary.

        Returns:
            Summary dictionary
        """
        return {
            "total_images": len(self.results),
            "detected": sum(1 for r in self.results if r.has_ear),
            "detection_rate": self.detection_rate,
            "average_ear_area": self.average_ear_area,
            "average_ear_percentage": np.mean(
                [r.ear_percentage for r in self.results if r.has_ear]
            )
            if any(r.has_ear for r in self.results)
            else 0.0,
        }
