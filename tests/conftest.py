"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import shutil
import tempfile
from typing import Generator

import cv2
import numpy as np
import pytest
import torch

from earsegmentationai.core.config import Config
from earsegmentationai.core.model import ModelManager


@pytest.fixture
def test_image() -> np.ndarray:
    """Create a test image."""
    # Create a simple test image with a white circle (simulating an ear)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 50, (255, 255, 255), -1)
    return image


@pytest.fixture
def test_mask() -> np.ndarray:
    """Create a test mask."""
    # Create a binary mask with a circle
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 50, 1, -1)
    return mask


@pytest.fixture
def test_batch_images() -> list:
    """Create a batch of test images."""
    images = []
    for i in range(4):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create circles at different positions
        x = 64 + i * 32
        y = 64 + i * 32
        cv2.circle(image, (x, y), 30, (255, 255, 255), -1)
        images.append(image)
    return images


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create test configuration."""
    config = Config(
        cache_dir=temp_dir / "cache",
        log_dir=temp_dir / "logs",
        processing={"device": "cpu", "batch_size": 2},
    )
    return config


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Return sigmoid-like output
            batch_size = x.shape[0]
            height = x.shape[2]
            width = x.shape[3]

            # Create circular mask output
            output = torch.zeros((batch_size, 1, height, width))
            center_y, center_x = height // 2, width // 2
            radius = min(height, width) // 4

            y, x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing="ij",
            )

            dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask = (dist < radius).float()

            for i in range(batch_size):
                output[i, 0] = mask

            return torch.sigmoid(output)

    return MockModel()


@pytest.fixture
def mock_model_manager(mock_model, test_config):
    """Create a mock model manager."""

    class MockModelManager(ModelManager):
        def __init__(self):
            super().__init__()
            self.config = test_config
            self._model = mock_model
            self._device = torch.device("cpu")
            self._initialized = True

        def load_model(self, force_download=False):
            return self._model

    return MockModelManager()


@pytest.fixture
def sample_image_files(temp_dir: Path) -> list:
    """Create sample image files."""
    image_dir = temp_dir / "images"
    image_dir.mkdir(exist_ok=True)

    files = []
    for i in range(3):
        # Create test image
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(image, (128, 128), 50 + i * 10, (255, 255, 255), -1)

        # Save image
        filename = f"test_image_{i}.png"
        filepath = image_dir / filename
        cv2.imwrite(str(filepath), image)
        files.append(filepath)

    return files


@pytest.fixture
def sample_video_file(temp_dir: Path) -> Path:
    """Create a sample video file."""
    video_path = temp_dir / "test_video.avi"

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (256, 256))

    # Write 30 frames
    for i in range(30):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        # Moving circle
        x = 128 + int(50 * np.sin(i * 0.2))
        y = 128 + int(50 * np.cos(i * 0.2))
        cv2.circle(frame, (x, y), 30, (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def disable_logging():
    """Disable logging during tests."""
    import logging

    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)
