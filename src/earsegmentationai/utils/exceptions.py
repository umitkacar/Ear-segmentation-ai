"""Custom exceptions for Ear Segmentation AI."""


class EarSegmentationError(Exception):
    """Base exception for all ear segmentation errors."""

    pass


class ModelError(EarSegmentationError):
    """Model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Model file not found error."""

    pass


class ModelLoadError(ModelError):
    """Error loading model."""

    pass


class InvalidModelError(ModelError):
    """Invalid model format or architecture."""

    pass


class ProcessingError(EarSegmentationError):
    """Processing-related errors."""

    pass


class InvalidInputError(ProcessingError):
    """Invalid input data error."""

    pass


class DeviceError(ProcessingError):
    """Device-related error (CPU/GPU)."""

    pass


class VideoError(ProcessingError):
    """Video processing error."""

    pass


class ConfigurationError(EarSegmentationError):
    """Configuration-related errors."""

    pass


class ValidationError(EarSegmentationError):
    """Input validation error."""

    pass
