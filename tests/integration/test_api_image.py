"""Integration tests for image processing API."""

import numpy as np
import pytest

from earsegmentationai.api.base import BatchProcessingResult, ProcessingResult
from earsegmentationai.api.image import ImageProcessor


class TestImageProcessorIntegration:
    """Integration tests for ImageProcessor."""

    def test_process_single_image_file(
        self, sample_image_files, mock_model_manager
    ):
        """Test processing a single image file."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        result = processor.process(sample_image_files[0])

        assert isinstance(result, ProcessingResult)
        assert result.has_ear  # Mock model always detects
        assert result.ear_percentage > 0
        assert result.metadata["source_path"] == str(sample_image_files[0])

    def test_process_numpy_array(self, test_image, mock_model_manager):
        """Test processing a numpy array."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        result = processor.process(test_image)

        assert isinstance(result, ProcessingResult)
        assert result.image.shape == test_image.shape
        assert result.mask.shape == test_image.shape[:2]

    def test_process_directory(self, sample_image_files, mock_model_manager):
        """Test processing a directory of images."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        directory = sample_image_files[0].parent
        result = processor.process(directory)

        assert isinstance(result, BatchProcessingResult)
        assert len(result) == 3  # We created 3 test images
        assert result.detection_rate > 0

    def test_process_list_of_images(
        self, sample_image_files, test_image, mock_model_manager
    ):
        """Test processing a list of mixed inputs."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        # Mix file paths and numpy arrays
        input_list = [
            sample_image_files[0],
            test_image,
            sample_image_files[1],
        ]

        result = processor.process(input_list)

        assert isinstance(result, BatchProcessingResult)
        assert len(result) == 3

    def test_return_probability_map(self, test_image, mock_model_manager):
        """Test returning probability map."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        result = processor.process(test_image, return_probability=True)

        assert result.probability_map is not None
        assert result.probability_map.shape == result.mask.shape
        assert result.probability_map.dtype == np.float32

    def test_return_visualization(self, test_image, mock_model_manager):
        """Test returning visualization."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        result = processor.process(test_image, return_visualization=True)

        assert "visualization" in result.metadata
        vis = result.metadata["visualization"]
        assert vis.shape == test_image.shape

    def test_save_results(self, test_image, temp_dir, mock_model_manager):
        """Test saving results to disk."""
        processor = ImageProcessor(model_manager=mock_model_manager)
        output_dir = temp_dir / "output"

        processor.process(
            test_image,
            save_results=True,
            output_dir=output_dir,
            return_probability=True,
            return_visualization=True,
        )

        # Check saved files
        assert output_dir.exists()
        assert (output_dir / "image_mask.png").exists()
        assert (output_dir / "image_probability.png").exists()
        assert (output_dir / "image_visualization.png").exists()

    def test_save_batch_results(
        self, sample_image_files, temp_dir, mock_model_manager
    ):
        """Test saving batch results."""
        processor = ImageProcessor(model_manager=mock_model_manager)
        output_dir = temp_dir / "batch_output"

        processor.process(
            sample_image_files, save_results=True, output_dir=output_dir
        )

        # Check saved files
        assert output_dir.exists()
        assert (output_dir / "summary.json").exists()

        # Check individual masks
        for i in range(len(sample_image_files)):
            # Filename includes .png extension
            assert (output_dir / f"test_image_{i}.png_mask.png").exists()

    def test_error_handling_invalid_path(self, mock_model_manager):
        """Test error handling for invalid path."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        with pytest.raises(Exception):
            processor.process("non_existent_file.jpg")

    def test_error_handling_empty_directory(
        self, temp_dir, mock_model_manager
    ):
        """Test handling empty directory."""
        processor = ImageProcessor(model_manager=mock_model_manager)
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = processor.process(empty_dir)

        assert isinstance(result, BatchProcessingResult)
        assert len(result) == 0

    def test_different_image_formats(self, temp_dir, mock_model_manager):
        """Test processing different image formats."""
        import cv2

        processor = ImageProcessor(model_manager=mock_model_manager)

        # Create test images in different formats
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        formats = [".jpg", ".png", ".bmp"]
        files = []

        for i, fmt in enumerate(formats):
            path = temp_dir / f"test{fmt}"
            cv2.imwrite(str(path), test_img)
            files.append(path)

        # Process all formats
        result = processor.process(files)

        assert len(result) == len(formats)
        assert all(r.has_ear for r in result)  # Mock always detects

    def test_processing_result_methods(self, test_image, mock_model_manager):
        """Test ProcessingResult methods."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        result = processor.process(test_image)

        # Test bounding box
        bbox = result.get_bounding_box()
        assert bbox is not None
        assert len(bbox) == 4  # x, y, w, h

        # Test center
        center = result.get_center()
        assert center is not None
        assert len(center) == 2  # x, y

        # Test to_dict
        result_dict = result.to_dict()
        assert "has_ear" in result_dict
        assert "ear_area" in result_dict
        assert "bounding_box" in result_dict

    def test_batch_result_methods(
        self, sample_image_files, mock_model_manager
    ):
        """Test BatchProcessingResult methods."""
        processor = ImageProcessor(model_manager=mock_model_manager)

        result = processor.process(sample_image_files)

        # Test filtering
        detected = result.filter_by_detection(detected=True)
        assert len(detected) > 0

        # Test summary
        summary = result.get_summary()
        assert "total_images" in summary
        assert "detection_rate" in summary
        assert summary["total_images"] == len(sample_image_files)
