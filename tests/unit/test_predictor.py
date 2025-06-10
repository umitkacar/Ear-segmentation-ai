"""Unit tests for predictor module."""

import numpy as np
import pytest

from earsegmentationai.core.predictor import EarPredictor, StreamPredictor
from earsegmentationai.utils.exceptions import ProcessingError


class TestEarPredictor:
    """Test EarPredictor class."""

    def test_initialization(self, mock_model_manager):
        """Test predictor initialization."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        assert predictor.threshold == 0.5
        assert predictor.model_manager is mock_model_manager

    def test_predict_single_image(self, mock_model_manager, test_image):
        """Test single image prediction."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        mask = predictor.predict(test_image)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8
        assert np.all(np.isin(mask, [0, 1]))

    def test_predict_with_probability(self, mock_model_manager, test_image):
        """Test prediction with probability map."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        mask, prob_map = predictor.predict(test_image, return_probability=True)

        assert isinstance(mask, np.ndarray)
        assert isinstance(prob_map, np.ndarray)
        assert mask.shape == prob_map.shape
        assert prob_map.dtype == np.float32
        assert np.all((prob_map >= 0) & (prob_map <= 1))

    def test_predict_batch(self, mock_model_manager, test_batch_images):
        """Test batch prediction."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        masks = predictor.predict_batch(test_batch_images, batch_size=2)

        assert len(masks) == len(test_batch_images)
        assert all(isinstance(m, np.ndarray) for m in masks)
        assert all(
            m.shape == img.shape[:2]
            for m, img in zip(masks, test_batch_images)
        )

    def test_predict_empty_batch(self, mock_model_manager):
        """Test prediction with empty batch."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        result = predictor.predict_batch([])

        assert result == []

    def test_set_threshold(self, mock_model_manager):
        """Test setting threshold."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        predictor.set_threshold(0.7)
        assert predictor.threshold == 0.7

        # Test invalid threshold
        with pytest.raises(ValueError):
            predictor.set_threshold(1.5)

        with pytest.raises(ValueError):
            predictor.set_threshold(-0.1)

    def test_warmup(self, mock_model_manager):
        """Test model warmup."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        # Should not raise error
        predictor.warmup(input_shape=(256, 256))

        # Test with default shape
        predictor.warmup()

    def test_resize_mask(self, mock_model_manager):
        """Test mask resizing."""
        predictor = EarPredictor(model_manager=mock_model_manager)

        # Create a small mask
        mask = np.ones((64, 64), dtype=np.float32)

        # Resize to larger
        resized = predictor._resize_mask(mask, (128, 128))

        assert resized.shape == (128, 128)

        # Test no resize needed
        same = predictor._resize_mask(mask, (64, 64))
        assert same.shape == mask.shape

    def test_error_handling(self, mock_model_manager, test_image):
        """Test error handling in prediction."""
        from unittest.mock import Mock

        predictor = EarPredictor(model_manager=mock_model_manager)

        # Mock model to raise error when called
        failing_model = Mock(side_effect=Exception("Model error"))
        mock_model_manager._model = failing_model

        with pytest.raises(ProcessingError, match="Failed to predict mask"):
            predictor.predict(test_image)


class TestStreamPredictor:
    """Test StreamPredictor class."""

    def test_initialization(self, mock_model_manager):
        """Test stream predictor initialization."""
        predictor = StreamPredictor(
            model_manager=mock_model_manager, skip_frames=2, smooth_masks=True
        )

        assert predictor.skip_frames == 2
        assert predictor.smooth_masks is True
        assert predictor.frame_count == 0
        assert predictor.last_mask is None

    def test_predict_frame_no_skip(self, mock_model_manager, test_image):
        """Test frame prediction without skipping."""
        predictor = StreamPredictor(
            model_manager=mock_model_manager, skip_frames=0
        )

        # Process multiple frames
        for i in range(3):
            mask = predictor.predict_frame(test_image)
            assert mask is not None
            assert mask.shape == test_image.shape[:2]
            assert predictor.frame_count == i + 1

    def test_predict_frame_with_skip(self, mock_model_manager, test_image):
        """Test frame prediction with skipping."""
        predictor = StreamPredictor(
            model_manager=mock_model_manager,
            skip_frames=2,  # Process every 3rd frame
        )

        results = []
        for i in range(6):
            mask = predictor.predict_frame(test_image)
            results.append(mask)

        # Should process frames 0, 3
        assert results[0] is not None  # Frame 0 - processed
        assert results[1] is not None  # Frame 1 - skipped, returns copy
        assert results[2] is not None  # Frame 2 - skipped, returns copy
        assert results[3] is not None  # Frame 3 - processed
        assert results[4] is not None  # Frame 4 - skipped
        assert results[5] is not None  # Frame 5 - skipped

    def test_predict_frame_with_probability(
        self, mock_model_manager, test_image
    ):
        """Test frame prediction with probability map."""
        predictor = StreamPredictor(
            model_manager=mock_model_manager, smooth_masks=True
        )

        mask, prob = predictor.predict_frame(
            test_image, return_probability=True
        )

        assert mask is not None
        assert prob is not None
        assert mask.shape == prob.shape

    def test_temporal_smoothing(self, mock_model_manager, test_image):
        """Test temporal smoothing between frames."""
        predictor = StreamPredictor(
            model_manager=mock_model_manager, smooth_masks=True
        )

        # Process multiple frames
        masks = []
        for _ in range(3):
            mask = predictor.predict_frame(test_image)
            masks.append(mask)

        # All masks should be valid
        assert all(m is not None for m in masks)

        # Last mask should be stored
        assert predictor.last_mask is not None

    def test_reset(self, mock_model_manager, test_image):
        """Test resetting stream state."""
        predictor = StreamPredictor(mock_model_manager)

        # Process a frame
        predictor.predict_frame(test_image)
        assert predictor.frame_count > 0
        assert predictor.last_mask is not None

        # Reset
        predictor.reset()

        assert predictor.frame_count == 0
        assert predictor.last_mask is None
        assert predictor.last_probability is None
