"""Unit tests for custom exceptions."""

import pytest

from earsegmentationai.utils.exceptions import (
    ConfigurationError,
    DeviceError,
    EarSegmentationError,
    InvalidInputError,
    InvalidModelError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    ProcessingError,
    ValidationError,
    VideoError,
)


class TestExceptions:
    """Test custom exception hierarchy and messages."""

    @pytest.mark.parametrize(
        "exc_cls,parent_cls",
        [
            (ModelError, EarSegmentationError),
            (ModelNotFoundError, ModelError),
            (ModelLoadError, ModelError),
            (InvalidModelError, ModelError),
            (ProcessingError, EarSegmentationError),
            (InvalidInputError, ProcessingError),
            (DeviceError, ProcessingError),
            (VideoError, ProcessingError),
            (ConfigurationError, EarSegmentationError),
            (ValidationError, EarSegmentationError),
        ],
    )
    def test_exception_inheritance(self, exc_cls, parent_cls):
        """Each custom exception should inherit from its parent."""
        assert issubclass(exc_cls, parent_cls)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            EarSegmentationError,
            ModelError,
            ModelNotFoundError,
            ModelLoadError,
            InvalidModelError,
            ProcessingError,
            InvalidInputError,
            DeviceError,
            VideoError,
            ConfigurationError,
            ValidationError,
        ],
    )
    def test_exception_message(self, exc_cls):
        """Custom exceptions should preserve the provided message."""
        message = "test error"
        with pytest.raises(exc_cls) as exc_info:
            raise exc_cls(message)
        assert str(exc_info.value) == message
