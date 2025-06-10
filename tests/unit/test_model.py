"""Unit tests for model module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from pathlib import Path

from earsegmentationai.core.model import ModelManager
from earsegmentationai.utils.exceptions import DeviceError, ModelLoadError


class TestModelManager:
    """Test ModelManager class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset ModelManager singleton before each test."""
        ModelManager._instance = None
        ModelManager._model = None
        ModelManager._device = None
        yield
        # Cleanup after test
        ModelManager._instance = None
        ModelManager._model = None
        ModelManager._device = None

    def test_singleton_pattern(self):
        """Test that ModelManager implements singleton pattern."""
        manager1 = ModelManager()
        manager2 = ModelManager()
        assert manager1 is manager2

    def test_device_selection_cpu(self, test_config):
        """Test CPU device selection."""
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            device = manager.device
            assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_fallback_to_cpu(self, mock_cuda, test_config):
        """Test fallback to CPU when CUDA not available."""
        test_config.processing.device = "cuda:0"
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            device = manager.device
            assert device.type == "cpu"

    def test_invalid_device_string(self, test_config):
        """Test invalid device string."""
        test_config.processing.device = "invalid:device"
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            with pytest.raises(DeviceError):
                _ = manager.device

    @patch("requests.get")
    def test_download_model(self, mock_get, test_config, temp_dir):
        """Test model download."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content = Mock(return_value=[b"data"] * 10)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            model_path = temp_dir / "test_model.pth"

            manager._download_model("http://example.com/model.pth", model_path)

            assert model_path.exists()
            mock_get.assert_called_once_with(
                "http://example.com/model.pth", stream=True
            )

    @patch("requests.get", side_effect=Exception("Network error"))
    def test_download_model_failure(self, mock_get, test_config, temp_dir):
        """Test model download failure."""
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            model_path = temp_dir / "test_model.pth"

            with pytest.raises(
                ModelLoadError, match="Failed to download model"
            ):
                manager._download_model(
                    "http://example.com/model.pth", model_path
                )

            # File should be cleaned up on failure
            assert not model_path.exists()

    def test_verify_model_exists(self, temp_dir):
        """Test model verification when file exists."""
        manager = ModelManager()
        model_path = temp_dir / "model.pth"
        model_path.write_bytes(b"model data")

        assert manager._verify_model(model_path) is True

    def test_verify_model_not_exists(self, temp_dir):
        """Test model verification when file doesn't exist."""
        manager = ModelManager()
        model_path = temp_dir / "non_existent.pth"

        assert manager._verify_model(model_path) is False

    @patch("segmentation_models_pytorch.Unet")
    def test_create_model(self, mock_unet_class, test_config):
        """Test model creation."""
        mock_model = Mock()
        mock_unet_class.return_value = mock_model

        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            model = manager._create_model()

            assert model is mock_model
            mock_unet_class.assert_called_once_with(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                classes=1,
                activation="sigmoid",
            )

    def test_get_model_info(self, test_config, mock_model):
        """Test getting model information."""
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            manager._model = mock_model
            manager._device = torch.device("cpu")

            info = manager.get_model_info()

            assert info["architecture"] == "Unet"
            assert info["encoder"] == "resnet18"
            assert info["device"] == "cpu"
            assert info["loaded"] is True

    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available", return_value=True)
    def test_clear_cache_gpu(
        self, mock_cuda_available, mock_empty_cache, test_config
    ):
        """Test clearing GPU cache."""
        test_config.processing.device = "cuda:0"
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            manager._device = torch.device("cuda:0")

            manager.clear_cache()

            mock_empty_cache.assert_called_once()

    def test_clear_cache_cpu(self, test_config):
        """Test clearing cache on CPU (no-op)."""
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            manager._device = torch.device("cpu")

            # Should not raise error
            manager.clear_cache()

    def test_set_device(self, test_config, mock_model):
        """Test changing device."""
        with patch(
            "earsegmentationai.core.model.get_config", return_value=test_config
        ):
            manager = ModelManager()
            manager._model = mock_model
            manager._device = torch.device("cpu")

            # Mock the to() method
            mock_model.to = Mock(return_value=mock_model)

            # Change device
            manager.set_device("cpu")  # Same device, but tests the logic

            assert manager._device.type == "cpu"
