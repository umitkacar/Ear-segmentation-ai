"""Visualization utilities for ear segmentation results."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)

# Default color palette
DEFAULT_COLORS = [
    (0, 0, 0),  # Background (black)
    (255, 0, 0),  # Ear (red)
    (0, 255, 0),  # Additional class (green)
    (0, 0, 255),  # Additional class (blue)
    (255, 255, 0),  # Additional class (yellow)
    (255, 0, 255),  # Additional class (magenta)
    (0, 255, 255),  # Additional class (cyan)
]


class MaskVisualizer:
    """Visualizer for segmentation masks."""

    def __init__(
        self,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        alpha: float = 0.5,
        show_contours: bool = True,
        contour_thickness: int = 2,
    ):
        """Initialize mask visualizer.

        Args:
            colors: Color palette for classes (BGR format)
            alpha: Transparency for overlay (0-1)
            show_contours: Whether to show mask contours
            contour_thickness: Thickness of contours
        """
        self.colors = colors or DEFAULT_COLORS
        self.alpha = alpha
        self.show_contours = show_contours
        self.contour_thickness = contour_thickness

    def visualize_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Visualize mask overlay on image.

        Args:
            image: Original image (H, W, C) in BGR format
            mask: Segmentation mask (H, W) with class indices
            class_names: Optional class names for legend

        Returns:
            Visualization image
        """
        # Ensure image is color
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Create overlay
        overlay = image.copy()

        # Apply colors for each class
        unique_classes = np.unique(mask)
        for class_idx in unique_classes:
            if class_idx == 0:  # Skip background
                continue

            if class_idx < len(self.colors):
                color = self.colors[class_idx]
            else:
                # Generate random color for unknown classes
                color = tuple(np.random.randint(0, 255, 3).tolist())

            # Create class mask
            class_mask = mask == class_idx
            overlay[class_mask] = color

        # Blend with original image
        result = cv2.addWeighted(image, 1 - self.alpha, overlay, self.alpha, 0)

        # Add contours if requested
        if self.show_contours:
            result = self._add_contours(result, mask)

        # Add legend if class names provided
        if class_names:
            result = self._add_legend(result, class_names, unique_classes)

        return result

    def visualize_probability(
        self,
        image: np.ndarray,
        probability_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Visualize probability map as heatmap.

        Args:
            image: Original image (H, W, C) in BGR format
            probability_map: Probability map (H, W) with values 0-1
            colormap: OpenCV colormap

        Returns:
            Visualization image
        """
        # Ensure image is color
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Convert probability to uint8
        prob_uint8 = (probability_map * 255).astype(np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(prob_uint8, colormap)

        # Blend with original image
        result = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

        return result

    def visualize_comparison(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Visualize prediction comparison with ground truth.

        Args:
            image: Original image
            mask: Predicted mask
            ground_truth: Ground truth mask (optional)

        Returns:
            Side-by-side comparison
        """
        # Original image
        vis_original = image.copy()

        # Prediction visualization
        vis_pred = self.visualize_mask(image, mask)

        if ground_truth is not None:
            # Ground truth visualization
            vis_gt = self.visualize_mask(image, ground_truth)

            # Difference visualization
            diff_mask = np.zeros_like(mask)
            diff_mask[
                (mask == 1) & (ground_truth == 0)
            ] = 1  # False positive (red)
            diff_mask[
                (mask == 0) & (ground_truth == 1)
            ] = 2  # False negative (blue)

            # Use custom colors for difference
            diff_visualizer = MaskVisualizer(
                colors=[(0, 0, 0), (0, 0, 255), (255, 0, 0)],
                alpha=0.7,
                show_contours=False,
            )
            vis_diff = diff_visualizer.visualize_mask(image, diff_mask)

            # Combine all visualizations
            row1 = np.hstack([vis_original, vis_pred])
            row2 = np.hstack([vis_gt, vis_diff])
            result = np.vstack([row1, row2])

            # Add labels
            result = self._add_text(result, "Original", (10, 30))
            result = self._add_text(
                result, "Prediction", (image.shape[1] + 10, 30)
            )
            result = self._add_text(
                result, "Ground Truth", (10, image.shape[0] + 30)
            )
            result = self._add_text(
                result,
                "Difference",
                (image.shape[1] + 10, image.shape[0] + 30),
            )
        else:
            # Just show original and prediction
            result = np.hstack([vis_original, vis_pred])
            result = self._add_text(result, "Original", (10, 30))
            result = self._add_text(
                result, "Prediction", (image.shape[1] + 10, 30)
            )

        return result

    def _add_contours(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add contours to visualization.

        Args:
            image: Image to draw on
            mask: Segmentation mask

        Returns:
            Image with contours
        """
        result = image.copy()

        # Find contours for each class
        unique_classes = np.unique(mask)
        for class_idx in unique_classes:
            if class_idx == 0:  # Skip background
                continue

            # Create binary mask for this class
            class_mask = (mask == class_idx).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw contours
            color = (
                self.colors[class_idx]
                if class_idx < len(self.colors)
                else (255, 255, 255)
            )
            cv2.drawContours(
                result, contours, -1, color, self.contour_thickness
            )

        return result

    def _add_legend(
        self,
        image: np.ndarray,
        class_names: List[str],
        class_indices: np.ndarray,
    ) -> np.ndarray:
        """Add legend to visualization.

        Args:
            image: Image to add legend to
            class_names: List of class names
            class_indices: Array of class indices present in the image

        Returns:
            Image with legend
        """
        result = image.copy()

        # Legend settings
        legend_height = 30
        legend_y = image.shape[0] - 10

        for i, class_idx in enumerate(class_indices):
            if class_idx == 0 or class_idx >= len(class_names):
                continue

            # Get color and name
            color = (
                self.colors[class_idx]
                if class_idx < len(self.colors)
                else (255, 255, 255)
            )
            name = class_names[class_idx]

            # Draw color box
            x = 10 + i * 150
            cv2.rectangle(
                result,
                (x, legend_y - legend_height),
                (x + 20, legend_y),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                result,
                name,
                (x + 25, legend_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return result

    def _add_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.8,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Add text to image.

        Args:
            image: Image to add text to
            text: Text string
            position: Text position (x, y)
            font_scale: Font scale
            color: Text color (BGR)
            thickness: Text thickness

        Returns:
            Image with text
        """
        # Add black background for better visibility
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        cv2.rectangle(
            image,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + 5),
            (0, 0, 0),
            -1,
        )

        # Add text
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

        return image


def create_grid_visualization(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    grid_shape: Optional[Tuple[int, int]] = None,
    image_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Create grid visualization of multiple results.

    Args:
        images: List of input images
        masks: List of predicted masks
        grid_shape: Grid shape (rows, cols). If None, auto-calculated.
        image_size: Size to resize each image to

    Returns:
        Grid visualization
    """
    if len(images) != len(masks):
        raise ValueError("Number of images and masks must match")

    n = len(images)

    # Auto-calculate grid shape if not provided
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid_shape = (rows, cols)

    # Create visualizer
    visualizer = MaskVisualizer()

    # Process each image
    vis_images = []
    for image, mask in zip(images, masks):
        # Resize image and mask
        image_resized = cv2.resize(image, image_size)
        mask_resized = cv2.resize(
            mask, image_size, interpolation=cv2.INTER_NEAREST
        )

        # Visualize
        vis = visualizer.visualize_mask(image_resized, mask_resized)
        vis_images.append(vis)

    # Pad with black images if needed
    total_cells = grid_shape[0] * grid_shape[1]
    while len(vis_images) < total_cells:
        black_image = np.zeros(
            (image_size[1], image_size[0], 3), dtype=np.uint8
        )
        vis_images.append(black_image)

    # Create grid
    rows = []
    for i in range(grid_shape[0]):
        row_images = vis_images[i * grid_shape[1] : (i + 1) * grid_shape[1]]
        row = np.hstack(row_images)
        rows.append(row)

    grid = np.vstack(rows)

    return grid
