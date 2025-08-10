"""Unit tests for visualization utilities."""

import numpy as np

from earsegmentationai.postprocessing.visualization import (
    MaskVisualizer,
    create_grid_visualization,
)


class TestVisualization:
    """Test visualization output shapes and types."""

    def test_visualize_mask_output(self):
        """visualize_mask should return an image with same shape and dtype."""
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:16, 8:16] = 1

        visualizer = MaskVisualizer(show_contours=False)
        result = visualizer.visualize_mask(image, mask)

        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_visualize_probability_output(self):
        """visualize_probability should match input image shape and dtype."""
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        probability = np.random.rand(32, 32).astype(np.float32)

        visualizer = MaskVisualizer(show_contours=False)
        result = visualizer.visualize_probability(image, probability)

        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_visualize_comparison_output(self):
        """visualize_comparison should produce grid with doubled dimensions."""
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        ground_truth = np.zeros((32, 32), dtype=np.uint8)

        visualizer = MaskVisualizer(show_contours=False)
        result = visualizer.visualize_comparison(image, mask, ground_truth)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_create_grid_visualization_output(self):
        """create_grid_visualization should return grid image with correct shape."""
        images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
        masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(4)]

        grid = create_grid_visualization(
            images, masks, grid_shape=(2, 2), image_size=(32, 32)
        )

        assert isinstance(grid, np.ndarray)
        assert grid.shape == (64, 64, 3)
        assert grid.dtype == np.uint8
