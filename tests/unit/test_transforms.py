"""Unit tests for image transformation module."""

import numpy as np
import pytest
import torch

from earsegmentationai.preprocessing.transforms import (
    BatchTransform,
    ImageTransform,
    create_augmentation_pipeline,
)


class TestImageTransform:
    """Test ImageTransform class."""

    def test_initialization(self):
        """Test transform initialization."""
        transform = ImageTransform(input_size=(256, 256))
        assert transform.input_size == (256, 256)
        assert transform.normalize is True
        assert transform.augment is False

    def test_basic_transform(self, test_image):
        """Test basic image transformation."""
        transform = ImageTransform(input_size=(128, 128))

        # Transform image
        result = transform(test_image)

        # Check output
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 128, 128)
        assert result.dtype == torch.float32

    def test_grayscale_input(self):
        """Test grayscale image handling."""
        # Create grayscale image
        gray_image = np.zeros((256, 256), dtype=np.uint8)

        transform = ImageTransform()
        result = transform(gray_image)

        # Should convert to RGB
        assert result.shape[0] == 3

    def test_rgba_input(self):
        """Test RGBA image handling."""
        # Create RGBA image
        rgba_image = np.zeros((256, 256, 4), dtype=np.uint8)

        transform = ImageTransform()
        result = transform(rgba_image)

        # Should convert to RGB
        assert result.shape[0] == 3

    def test_with_mask(self, test_image, test_mask):
        """Test transformation with mask."""
        transform = ImageTransform(input_size=(128, 128))

        # Transform image and mask
        image_result, mask_result = transform(test_image, test_mask)

        # Check outputs
        assert isinstance(image_result, torch.Tensor)
        assert isinstance(mask_result, torch.Tensor)
        assert image_result.shape == (3, 128, 128)
        assert mask_result.shape == (128, 128)

    def test_no_normalization(self, test_image):
        """Test transformation without normalization."""
        transform = ImageTransform(normalize=False)
        result = transform(test_image)

        # ToTensorV2 preserves the dtype when no normalization is applied
        # The test image has white circle with value 255
        assert result.min() >= 0
        if result.dtype == torch.float32:
            assert result.max() <= 1.0
        else:
            assert result.dtype == torch.uint8
            assert result.max() <= 255

    def test_with_augmentation(self, test_image):
        """Test transformation with augmentation."""
        transform = ImageTransform(augment=True)

        # Multiple transforms should give different results (probabilistic)
        results = [transform(test_image) for _ in range(5)]

        # At least some should be different due to augmentation
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same  # With 5 samples, very unlikely all are the same

    def test_inverse_transform(self, test_image):
        """Test inverse transformation."""
        transform = ImageTransform()

        # Forward transform
        tensor = transform(test_image)

        # Inverse transform
        recovered = transform.inverse_transform(tensor)

        # Check output
        assert isinstance(recovered, np.ndarray)
        # Note: inverse transform doesn't reverse the resize operation
        assert recovered.shape == (
            transform.input_size[1],
            transform.input_size[0],
            3,
        )
        assert recovered.dtype == np.uint8

    def test_inverse_transform_batch(self, test_image):
        """Test inverse transform with batch dimension."""
        transform = ImageTransform()

        # Create batch tensor
        tensor = transform(test_image)
        batch_tensor = tensor.unsqueeze(0)  # Add batch dimension

        # Inverse transform
        recovered = transform.inverse_transform(batch_tensor)

        # Should handle batch dimension and return single image
        assert recovered.shape == (
            transform.input_size[1],
            transform.input_size[0],
            3,
        )


class TestBatchTransform:
    """Test BatchTransform class."""

    def test_initialization(self):
        """Test batch transform initialization."""
        batch_transform = BatchTransform()
        assert isinstance(batch_transform.transform, ImageTransform)

    def test_batch_processing(self, test_batch_images):
        """Test batch image processing."""
        batch_transform = BatchTransform()

        # Process batch
        result = batch_transform(test_batch_images)

        # Check output
        assert isinstance(result, torch.Tensor)
        # Default size is (width=480, height=320), but tensor shape is (batch, channels, height, width)
        assert result.shape == (4, 3, 320, 480)

    def test_custom_transform(self, test_batch_images):
        """Test batch transform with custom image transform."""
        image_transform = ImageTransform(input_size=(128, 128))
        batch_transform = BatchTransform(image_transform)

        # Process batch
        result = batch_transform(test_batch_images)

        # Check output with custom size
        assert result.shape == (4, 3, 128, 128)

    def test_collate_fn(self, test_batch_images):
        """Test collate function for DataLoader."""
        batch_transform = BatchTransform()

        # Create batch with metadata
        batch_data = [
            (img, {"id": i}) for i, img in enumerate(test_batch_images)
        ]

        # Collate
        images, metadata = batch_transform.collate_fn(batch_data)

        # Check outputs
        assert isinstance(images, torch.Tensor)
        assert len(metadata) == 4
        assert all(m["id"] == i for i, m in enumerate(metadata))


class TestAugmentationPipeline:
    """Test augmentation pipeline creation."""

    def test_create_light_augmentation(self):
        """Test light augmentation pipeline."""
        pipeline = create_augmentation_pipeline((256, 256), strength="light")

        # Check it's a valid albumentations Compose object
        assert hasattr(pipeline, "transforms")

        # Test on image
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = pipeline(image=image)

        assert "image" in result
        assert isinstance(result["image"], torch.Tensor)

    def test_create_medium_augmentation(self):
        """Test medium augmentation pipeline."""
        pipeline = create_augmentation_pipeline((256, 256), strength="medium")

        # Test probability values are correct
        for t in pipeline.transforms:
            if hasattr(t, "p"):
                assert t.p in [0.3, 0.5, 1.0]  # Expected probabilities

    def test_create_heavy_augmentation(self):
        """Test heavy augmentation pipeline."""
        pipeline = create_augmentation_pipeline((256, 256), strength="heavy")

        # Test probability values are correct
        for t in pipeline.transforms:
            if hasattr(t, "p"):
                assert t.p in [0.5, 0.7, 1.0]  # Expected probabilities

    def test_invalid_strength(self):
        """Test invalid augmentation strength."""
        with pytest.raises(ValueError, match="Unknown augmentation strength"):
            create_augmentation_pipeline((256, 256), strength="invalid")

    def test_augmentation_with_mask(self, test_image, test_mask):
        """Test augmentation pipeline with mask."""
        pipeline = create_augmentation_pipeline((256, 256), strength="medium")

        # Apply to image and mask
        result = pipeline(image=test_image, mask=test_mask)

        assert "image" in result
        assert "mask" in result
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["mask"], torch.Tensor)
