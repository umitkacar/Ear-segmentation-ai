"""Unit tests for input validation module."""


import numpy as np
import pytest

from earsegmentationai.preprocessing.validators import (
    get_image_files,
    validate_batch_size,
    validate_camera_id,
    validate_device_string,
    validate_directory,
    validate_image_array,
    validate_image_path,
    validate_mask_array,
    validate_video_path,
)
from earsegmentationai.utils.exceptions import (
    InvalidInputError,
    ValidationError,
)


class TestImagePathValidation:
    """Test image path validation."""

    def test_valid_image_path(self, sample_image_files):
        """Test validation of valid image path."""
        path = sample_image_files[0]
        result = validate_image_path(path)
        assert result == path

    def test_invalid_image_path(self):
        """Test validation of invalid image path."""
        with pytest.raises(ValidationError, match="Image file not found"):
            validate_image_path("non_existent.jpg")

    def test_directory_as_image_path(self, temp_dir):
        """Test validation when directory is passed as image."""
        with pytest.raises(ValidationError, match="Path is not a file"):
            validate_image_path(temp_dir)

    def test_unsupported_image_format(self, temp_dir):
        """Test validation of unsupported image format."""
        # Create a text file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test")

        with pytest.raises(ValidationError, match="Unsupported image format"):
            validate_image_path(txt_file)


class TestVideoPathValidation:
    """Test video path validation."""

    def test_valid_video_path(self, sample_video_file):
        """Test validation of valid video path."""
        result = validate_video_path(sample_video_file)
        assert result == sample_video_file

    def test_invalid_video_path(self):
        """Test validation of invalid video path."""
        with pytest.raises(ValidationError, match="Video file not found"):
            validate_video_path("non_existent.mp4")

    def test_unsupported_video_format(self, temp_dir):
        """Test validation of unsupported video format."""
        # Create a text file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test")

        with pytest.raises(ValidationError, match="Unsupported video format"):
            validate_video_path(txt_file)


class TestDirectoryValidation:
    """Test directory validation."""

    def test_valid_directory(self, temp_dir):
        """Test validation of valid directory."""
        result = validate_directory(temp_dir)
        assert result == temp_dir

    def test_create_directory(self, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "new_directory"
        result = validate_directory(new_dir, create=True)

        assert result == new_dir
        assert new_dir.exists()

    def test_non_existent_directory(self):
        """Test validation of non-existent directory."""
        with pytest.raises(ValidationError, match="Directory not found"):
            validate_directory("non_existent_dir", create=False)

    def test_file_as_directory(self, sample_image_files):
        """Test validation when file is passed as directory."""
        with pytest.raises(ValidationError, match="Path is not a directory"):
            validate_directory(sample_image_files[0])


class TestImageArrayValidation:
    """Test image array validation."""

    def test_valid_image_array(self, test_image):
        """Test validation of valid image array."""
        result = validate_image_array(test_image)
        assert np.array_equal(result, test_image)

    def test_invalid_type(self):
        """Test validation with invalid type."""
        with pytest.raises(
            InvalidInputError, match="Image must be numpy array"
        ):
            validate_image_array([1, 2, 3])

    def test_empty_array(self):
        """Test validation of empty array."""
        with pytest.raises(InvalidInputError, match="Image is empty"):
            validate_image_array(np.array([]))

    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        # 1D array
        with pytest.raises(InvalidInputError, match="Image must be 2D or 3D"):
            validate_image_array(np.ones(10))

        # 4D array
        with pytest.raises(InvalidInputError, match="Image must be 2D or 3D"):
            validate_image_array(np.ones((10, 10, 10, 10)))

    def test_invalid_channels(self):
        """Test validation of invalid channel count."""
        with pytest.raises(
            InvalidInputError, match="Image must have 1, 3, or 4 channels"
        ):
            validate_image_array(np.ones((100, 100, 2)))


class TestMaskArrayValidation:
    """Test mask array validation."""

    def test_valid_binary_mask(self, test_mask):
        """Test validation of valid binary mask."""
        result = validate_mask_array(test_mask, num_classes=1)
        assert np.array_equal(result, test_mask)

    def test_invalid_type(self):
        """Test validation with invalid type."""
        with pytest.raises(
            InvalidInputError, match="Mask must be numpy array"
        ):
            validate_mask_array([1, 2, 3])

    def test_empty_mask(self):
        """Test validation of empty mask."""
        with pytest.raises(InvalidInputError, match="Mask is empty"):
            validate_mask_array(np.array([]))

    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(InvalidInputError, match="Mask must be 2D or 3D"):
            validate_mask_array(np.ones((10, 10, 10, 10)))

    def test_multi_class_mask(self):
        """Test validation of multi-class mask."""
        # Create mask with 3 classes (0, 1, 2)
        mask = np.random.randint(0, 3, (100, 100))
        result = validate_mask_array(mask, num_classes=3)
        assert np.array_equal(result, mask)

    def test_invalid_class_values(self):
        """Test validation with invalid class values."""
        # Mask with class 3, but only 3 classes expected (0, 1, 2)
        mask = np.ones((100, 100)) * 3
        with pytest.raises(InvalidInputError, match="Mask contains class 3"):
            validate_mask_array(mask, num_classes=3)


class TestDeviceStringValidation:
    """Test device string validation."""

    def test_cpu_device(self):
        """Test CPU device validation."""
        result = validate_device_string("cpu")
        assert result == "cpu"

    def test_cuda_device(self):
        """Test CUDA device validation."""
        import torch

        if torch.cuda.is_available():
            result = validate_device_string("cuda:0")
            assert result == "cuda:0"
        else:
            # Should fallback to CPU
            result = validate_device_string("cuda:0")
            assert result == "cpu"

    def test_invalid_cuda_format(self):
        """Test invalid CUDA device format."""
        with pytest.raises(
            ValidationError, match="Invalid CUDA device format"
        ):
            validate_device_string("cuda:abc")

    def test_invalid_device(self):
        """Test completely invalid device string."""
        with pytest.raises(ValidationError, match="Unknown device"):
            validate_device_string("tpu")


class TestBatchSizeValidation:
    """Test batch size validation."""

    def test_valid_batch_size(self):
        """Test valid batch size."""
        result = validate_batch_size(16)
        assert result == 16

    def test_invalid_type(self):
        """Test invalid batch size type."""
        with pytest.raises(
            ValidationError, match="Batch size must be integer"
        ):
            validate_batch_size(16.5)

    def test_negative_batch_size(self):
        """Test negative batch size."""
        with pytest.raises(
            ValidationError, match="Batch size must be positive"
        ):
            validate_batch_size(-1)

    def test_zero_batch_size(self):
        """Test zero batch size."""
        with pytest.raises(
            ValidationError, match="Batch size must be positive"
        ):
            validate_batch_size(0)


class TestCameraIdValidation:
    """Test camera ID validation."""

    def test_valid_integer_camera_id(self):
        """Test valid integer camera ID."""
        from unittest.mock import Mock, patch

        # Mock cv2.VideoCapture to simulate camera availability
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.release = Mock()

        with patch(
            "earsegmentationai.preprocessing.validators.cv2.VideoCapture",
            return_value=mock_cap,
        ):
            result = validate_camera_id(0)
            assert result == 0

        # Verify VideoCapture was called and released
        mock_cap.release.assert_called_once()

    def test_negative_camera_id(self):
        """Test negative camera ID."""
        with pytest.raises(
            ValidationError, match="Camera ID must be non-negative"
        ):
            validate_camera_id(-1)

    def test_device_path_camera_id(self):
        """Test device path as camera ID."""
        # This won't actually exist on most systems, but tests the logic
        with pytest.raises(ValidationError, match="Camera device not found"):
            validate_camera_id("/dev/video99")

    def test_url_camera_id(self):
        """Test URL as camera ID."""
        # Test that URLs are accepted (actual connection not tested)
        result = validate_camera_id("http://example.com/stream")
        assert result == "http://example.com/stream"

        result = validate_camera_id("rtsp://example.com/stream")
        assert result == "rtsp://example.com/stream"

    def test_invalid_camera_id_type(self):
        """Test invalid camera ID type."""
        with pytest.raises(
            ValidationError, match="Camera ID must be integer or string"
        ):
            validate_camera_id(12.5)


class TestGetImageFiles:
    """Test get_image_files function."""

    def test_get_image_files(self, sample_image_files):
        """Test getting image files from directory."""
        directory = sample_image_files[0].parent
        files = get_image_files(directory)

        assert len(files) == 3
        assert all(f.suffix == ".png" for f in files)

    def test_empty_directory(self, temp_dir):
        """Test getting files from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        files = get_image_files(empty_dir)
        assert files == []

    def test_mixed_files(self, temp_dir):
        """Test directory with mixed file types."""
        # Create image and non-image files
        (temp_dir / "image.jpg").touch()
        (temp_dir / "image.PNG").touch()  # Test case sensitivity
        (temp_dir / "document.txt").touch()
        (temp_dir / "data.csv").touch()

        files = get_image_files(temp_dir)

        assert len(files) == 2
        assert any(f.name == "image.jpg" for f in files)
        assert any(f.name == "image.PNG" for f in files)
