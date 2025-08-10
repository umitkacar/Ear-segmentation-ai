"""Unit tests for logging utilities."""

import logging
from unittest.mock import patch

from earsegmentationai.utils.logging import get_logger, setup_logging


class TestLogging:
    """Test logging setup and retrieval."""

    def test_setup_logging_level(self, caplog, test_config):
        """setup_logging should respect the configured level."""
        with patch(
            "earsegmentationai.utils.logging.get_config", return_value=test_config
        ):
            logger = setup_logging(name="test_logger", level="INFO", use_rich=False)

        caplog.set_level(logging.DEBUG, logger="test_logger")
        logger.debug("debug message")
        logger.info("info message")

        messages = [r.message for r in caplog.records if r.name == "test_logger"]
        assert "info message" in messages
        assert "debug message" not in messages

    def test_get_logger_level(self, caplog, test_config):
        """get_logger should return logger with previously set level."""
        with patch(
            "earsegmentationai.utils.logging.get_config", return_value=test_config
        ):
            setup_logging(name="existing_logger", level="WARNING", use_rich=False)

        logger = get_logger("existing_logger")
        caplog.set_level(logging.DEBUG, logger="existing_logger")

        logger.info("info message")
        logger.error("error message")

        messages = [r.message for r in caplog.records if r.name == "existing_logger"]
        assert "error message" in messages
        assert "info message" not in messages
