"""Prediction logic for ear segmentation."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import cv2

from earsegmentationai.core.config import get_config
from earsegmentationai.core.model import ModelManager
from earsegmentationai.preprocessing.transforms import (
    BatchTransform,
    ImageTransform,
)
from earsegmentationai.preprocessing.validators import validate_image_array
from earsegmentationai.utils.exceptions import ProcessingError
from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)


class EarPredictor:
    """Ear segmentation predictor.

    This class handles the prediction logic for ear segmentation,
    including preprocessing, inference, and postprocessing.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        transform: Optional[ImageTransform] = None,
        threshold: float = 0.5,
    ):
        """Initialize predictor.

        Args:
            model_manager: Model manager instance. If None, creates new one.
            transform: Image transform. If None, creates default.
            threshold: Binary threshold for mask generation.
        """
        self.model_manager = model_manager or ModelManager()
        self.transform = transform or ImageTransform()
        self.batch_transform = BatchTransform(self.transform)
        self.threshold = threshold
        self.config = get_config()

        logger.info(f"EarPredictor initialized with threshold={threshold}")

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_probability: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict ear mask for a single image.

        Args:
            image: Input image (H, W, C)
            return_probability: Whether to return probability map

        Returns:
            Binary mask (H, W) or tuple of (mask, probability_map)

        Raises:
            ProcessingError: If prediction fails
        """
        try:
            # Validate input
            image = validate_image_array(image)
            original_shape = image.shape[:2]

            # Transform image
            tensor = self.transform(image)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # Add batch dimension

            # Move to device
            tensor = tensor.to(self.model_manager.device)

            # Run inference
            model = self.model_manager.model
            model.eval()

            output = model(tensor)

            # Apply sigmoid if needed
            if self.config.model.activation != "sigmoid":
                output = torch.sigmoid(output)

            # Convert to numpy
            probability_map = output.squeeze(0).squeeze(0).cpu().numpy()

            # Resize to original shape
            probability_map = self._resize_mask(
                probability_map, original_shape
            )

            # Generate binary mask
            binary_mask = (probability_map > self.threshold).astype(np.uint8)

            if return_probability:
                return binary_mask, probability_map
            else:
                return binary_mask

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ProcessingError(f"Failed to predict mask: {e}")

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: Optional[int] = None,
        return_probability: bool = False,
    ) -> Union[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """Predict ear masks for multiple images.

        Args:
            images: List of input images
            batch_size: Batch size for processing. If None, uses config default.
            return_probability: Whether to return probability maps

        Returns:
            List of binary masks or list of (mask, probability_map) tuples

        Raises:
            ProcessingError: If prediction fails
        """
        try:
            if not images:
                return []

            batch_size = batch_size or self.config.processing.batch_size
            results = []

            # Process in batches
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_results = self._predict_batch(
                    batch_images, return_probability
                )
                results.extend(batch_results)

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise ProcessingError(f"Failed to predict batch: {e}")

    def _predict_batch(
        self,
        images: List[np.ndarray],
        return_probability: bool = False,
    ) -> Union[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """Internal batch prediction method.

        Args:
            images: List of input images
            return_probability: Whether to return probability maps

        Returns:
            List of results
        """
        # Validate and store original shapes
        validated_images = []
        original_shapes = []

        for image in images:
            image = validate_image_array(image)
            validated_images.append(image)
            original_shapes.append(image.shape[:2])

        # Transform batch
        batch_tensor = self.batch_transform(validated_images)
        batch_tensor = batch_tensor.to(self.model_manager.device)

        # Run inference
        model = self.model_manager.model
        model.eval()

        outputs = model(batch_tensor)

        # Apply sigmoid if needed
        if self.config.model.activation != "sigmoid":
            outputs = torch.sigmoid(outputs)

        # Process each output
        results = []
        for i, output in enumerate(outputs):
            # Convert to numpy
            probability_map = output.squeeze(0).cpu().numpy()

            # Resize to original shape
            probability_map = self._resize_mask(
                probability_map, original_shapes[i]
            )

            # Generate binary mask
            binary_mask = (probability_map > self.threshold).astype(np.uint8)

            if return_probability:
                results.append((binary_mask, probability_map))
            else:
                results.append(binary_mask)

        return results

    def _resize_mask(
        self,
        mask: np.ndarray,
        target_shape: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Resize mask to target shape.

        Args:
            mask: Input mask
            target_shape: Target shape (H, W)
            interpolation: Interpolation method

        Returns:
            Resized mask
        """

        if mask.shape == target_shape:
            return mask

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Resize
        resized = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),  # cv2 uses (W, H)
            interpolation=interpolation,
        )

        return resized

    def set_threshold(self, threshold: float) -> None:
        """Set binary threshold.

        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"Threshold must be between 0 and 1, got {threshold}"
            )

        self.threshold = threshold
        logger.info(f"Threshold set to {threshold}")

    def warmup(self, input_shape: Optional[Tuple[int, int]] = None) -> None:
        """Warmup model with dummy input.

        Args:
            input_shape: Input shape (H, W). If None, uses config default.
        """
        try:
            config = get_config()
            if input_shape is None:
                input_shape = (
                    config.processing.input_size[1],
                    config.processing.input_size[0],
                )

            # Create dummy input
            dummy_image = np.zeros((*input_shape, 3), dtype=np.uint8)

            # Run prediction
            logger.info("Warming up model...")
            self.predict(dummy_image)
            logger.info("Model warmup complete")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")


class StreamPredictor(EarPredictor):
    """Predictor optimized for video streams.

    This predictor includes additional optimizations for real-time
    video processing, such as frame skipping and temporal smoothing.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        transform: Optional[ImageTransform] = None,
        threshold: float = 0.5,
        skip_frames: int = 0,
        smooth_masks: bool = True,
    ):
        """Initialize stream predictor.

        Args:
            model_manager: Model manager instance
            transform: Image transform
            threshold: Binary threshold
            skip_frames: Number of frames to skip between predictions
            smooth_masks: Whether to apply temporal smoothing
        """
        super().__init__(model_manager, transform, threshold)
        self.skip_frames = skip_frames
        self.smooth_masks = smooth_masks
        self.frame_count = 0
        self.last_mask = None
        self.last_probability = None

    def predict_frame(
        self,
        frame: np.ndarray,
        return_probability: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]:
        """Predict mask for a single video frame.

        Args:
            frame: Input frame
            return_probability: Whether to return probability map

        Returns:
            Mask, (mask, probability), or None if frame is skipped
        """
        # Check if we should skip this frame
        if (
            self.skip_frames > 0
            and self.frame_count % (self.skip_frames + 1) != 0
        ):
            self.frame_count += 1
            if self.last_mask is not None:
                if return_probability and self.last_probability is not None:
                    return self.last_mask.copy(), self.last_probability.copy()
                else:
                    return self.last_mask.copy()
            return None

        # Run prediction
        if return_probability:
            mask, probability = self.predict(frame, return_probability=True)

            # Apply temporal smoothing
            if self.smooth_masks and self.last_probability is not None:
                alpha = 0.7  # Weight for current frame
                probability = (
                    alpha * probability + (1 - alpha) * self.last_probability
                )
                mask = (probability > self.threshold).astype(np.uint8)

            self.last_mask = mask
            self.last_probability = probability
            self.frame_count += 1

            return mask, probability
        else:
            mask = self.predict(frame, return_probability=False)

            # Apply temporal smoothing
            if self.smooth_masks and self.last_mask is not None:
                # Simple morphological smoothing
                kernel = np.ones((3, 3), np.uint8)
                mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                last_dilated = cv2.dilate(self.last_mask, kernel, iterations=1)
                mask = cv2.bitwise_and(mask_dilated, last_dilated)

            self.last_mask = mask
            self.frame_count += 1

            return mask

    def reset(self) -> None:
        """Reset stream state."""
        self.frame_count = 0
        self.last_mask = None
        self.last_probability = None
