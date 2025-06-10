"""Image transformation utilities for preprocessing."""

from typing import List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from earsegmentationai.core.config import get_config
from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)


class ImageTransform:
    """Image transformation pipeline for ear segmentation.
    
    This class handles all image preprocessing steps including resizing,
    normalization, and augmentation.
    """
    
    def __init__(
        self,
        input_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        augment: bool = False,
    ):
        """Initialize image transform.
        
        Args:
            input_size: Target size (width, height). If None, uses config default.
            normalize: Whether to normalize images
            augment: Whether to apply augmentations (for training)
        """
        config = get_config()
        self.input_size = input_size or config.processing.input_size
        self.normalize = normalize
        self.augment = augment
        
        self.transform = self._build_transform()
        logger.debug(f"ImageTransform initialized with size {self.input_size}")
    
    def _build_transform(self) -> A.Compose:
        """Build transformation pipeline.
        
        Returns:
            Albumentations composition
        """
        transforms = []
        
        # Resize
        transforms.append(
            A.Resize(
                height=self.input_size[1],
                width=self.input_size[0],
                interpolation=cv2.INTER_LINEAR,
                always_apply=True,
            )
        )
        
        # Augmentations (only for training)
        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
            ])
        
        # Normalization
        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True,
                )
            )
        
        # Convert to tensor
        transforms.append(ToTensorV2(always_apply=True))
        
        return A.Compose(transforms)
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply transformations to image and optionally mask.
        
        Args:
            image: Input image (H, W, C) in BGR or RGB format
            mask: Optional mask (H, W)
            
        Returns:
            Transformed image tensor or tuple of (image, mask) tensors
        """
        # Ensure image is in the right format
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Assume BGR format from OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            return transformed["image"], transformed["mask"]
        else:
            transformed = self.transform(image=image)
            return transformed["image"]
    
    def inverse_transform(self, tensor: torch.Tensor) -> np.ndarray:
        """Inverse transform tensor back to image.
        
        Args:
            tensor: Image tensor (C, H, W)
            
        Returns:
            Image array (H, W, C) in RGB format
        """
        # Move to CPU and convert to numpy
        image = tensor.cpu().numpy()
        
        # Handle batch dimension
        if image.ndim == 4:
            image = image[0]
        
        # Transpose from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        
        # Denormalize if needed
        if self.normalize and image.shape[2] == 3:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
        
        # Clip values and convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image


class BatchTransform:
    """Batch transformation for multiple images."""
    
    def __init__(self, transform: Optional[ImageTransform] = None):
        """Initialize batch transform.
        
        Args:
            transform: Image transform to use. If None, creates default.
        """
        self.transform = transform or ImageTransform()
    
    def __call__(self, images: List[np.ndarray]) -> torch.Tensor:
        """Transform batch of images.
        
        Args:
            images: List of images
            
        Returns:
            Batch tensor (B, C, H, W)
        """
        tensors = [self.transform(image) for image in images]
        return torch.stack(tensors)
    
    def collate_fn(self, batch: List[Tuple[np.ndarray, any]]) -> Tuple[torch.Tensor, List[any]]:
        """Collate function for DataLoader.
        
        Args:
            batch: List of (image, metadata) tuples
            
        Returns:
            Tuple of (image_batch, metadata_list)
        """
        images, metadata = zip(*batch)
        image_batch = self(list(images))
        return image_batch, list(metadata)


def create_augmentation_pipeline(
    input_size: Tuple[int, int],
    strength: str = "medium"
) -> A.Compose:
    """Create augmentation pipeline for training.
    
    Args:
        input_size: Target size (width, height)
        strength: Augmentation strength ("light", "medium", "heavy")
        
    Returns:
        Albumentations composition
    """
    if strength == "light":
        p_spatial = 0.3
        p_pixel = 0.2
    elif strength == "medium":
        p_spatial = 0.5
        p_pixel = 0.3
    elif strength == "heavy":
        p_spatial = 0.7
        p_pixel = 0.5
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")
    
    return A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=p_spatial),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=p_spatial
        ),
        A.RandomResizedCrop(
            height=input_size[1],
            width=input_size[0],
            scale=(0.8, 1.0),
            p=p_spatial
        ),
        
        # Pixel augmentations
        A.RandomBrightnessContrast(p=p_pixel),
        A.RandomGamma(p=p_pixel),
        A.HueSaturationValue(p=p_pixel),
        A.CLAHE(p=p_pixel),
        
        # Noise and blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
            A.MotionBlur(blur_limit=5),
        ], p=p_pixel),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(),
        ], p=p_pixel),
        
        # Final resize and normalize
        A.Resize(height=input_size[1], width=input_size[0]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])