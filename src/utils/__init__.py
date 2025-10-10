"""
Utility modules initialization.
"""

from .config import settings, get_settings, load_config, is_feature_enabled
from .logger import get_logger, setup_logging, LoggerMixin, init_logging_from_config
from .exceptions import *
from .helpers import *

# Initialize logging when module is imported
init_logging_from_config()

__all__ = [
    # Config
    'settings',
    'get_settings', 
    'load_config',
    'is_feature_enabled',
    
    # Logging
    'get_logger',
    'setup_logging',
    'LoggerMixin',
    
    # Exceptions
    'RCMSException',
    'ConfigurationError',
    'ModelError',
    'ModelNotFoundError',
    'ModelLoadError',
    'PredictionError',
    'ValidationError',
    'FileProcessingError',
    'UnsupportedFileTypeError',
    'FileSizeExceededError',
    'DatabaseError',
    'AuthenticationError',
    'AuthorizationError',
    'APIError',
    'ExternalServiceError',
    'PreprocessingError',
    'OCRError',
    'ASRError',
    'ComputerVisionError',
    'SentimentAnalysisError',
    
    # Helpers
    'get_file_hash',
    'get_file_type',
    'is_image_file',
    'is_audio_file', 
    'is_video_file',
    'get_file_size_mb',
    'ensure_directory',
    'clean_text',
    'validate_file_extension',
    'generate_unique_filename',
    'safe_dict_get',
    'format_bytes',
    'truncate_text',
]