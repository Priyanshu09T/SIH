"""
=====================================================
Logging utilities for RCMS (Railway Complaint Management System)
=====================================================

This module provides structured logging with automatic fallback to Python's 
standard `logging` library if `structlog` is unavailable.

Features:
---------
- JSON structured logs (if structlog available)
- Console + File logging support
- Configurable log level and file path
- Safe fallback and config integration

Usage:
------
from src.utils.logger import get_logger, init_logging_from_config

logger = get_logger(__name__)
logger.info("Application started")
"""

import logging
from pathlib import Path
from typing import Optional, Union, Any

# Try to import structlog
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
    LoggerType = structlog.BoundLogger
except ImportError:
    STRUCTLOG_AVAILABLE = False
    LoggerType = logging.Logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup application logging configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Optional log file path
    """

    # Configure structlog if available
    if STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    # Configure standard logging
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True  # Overrides existing logging config
    )


def get_logger(name: str) -> Union[Any, logging.Logger]:
    """
    Get a structured logger instance.

    Args:
        name (str): Logger name (usually __name__)

    Returns:
        Configured logger (structlog if available, otherwise standard logger)
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> Union[Any, logging.Logger]:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def init_logging_from_config():
    """
    Initialize logging from configuration file.
    Falls back to default configuration if config fails or missing.
    """
    try:
        from src.utils.config import load_config
        config = load_config()
        logging_config = config.get('logging', {})

        log_level = logging_config.get('level', 'INFO')
        log_file = logging_config.get('file')

        setup_logging(log_level, log_file)

    except Exception as e:
        # Fallback to basic logging
        setup_logging()
        logger = get_logger(__name__)
        if hasattr(logger, 'warning'):
            logger.warning(f"⚠️ Failed to load logging config, using defaults: {e}")
        else:
            print(f"Warning: Failed to load logging config, using defaults: {e}")


# Initialize basic logging for immediate use (optional)
setup_logging()
