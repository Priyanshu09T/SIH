"""
Common exceptions for RCMS application.
Defines custom exception classes for better error handling.
"""

from typing import Optional, Any, Dict


class RCMSException(Exception):
    """Base exception for RCMS application."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(RCMSException):
    """Raised when there's an error in configuration."""
    pass


class ModelError(RCMSException):
    """Base exception for ML model related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a model file is not found."""
    pass


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    pass


class PredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class ValidationError(RCMSException):
    """Raised when input validation fails."""
    pass


class FileProcessingError(RCMSException):
    """Raised when file processing fails."""
    pass


class UnsupportedFileTypeError(FileProcessingError):
    """Raised when an unsupported file type is encountered."""
    pass


class FileSizeExceededError(FileProcessingError):
    """Raised when uploaded file exceeds size limit."""
    pass


class DatabaseError(RCMSException):
    """Raised when database operations fail."""
    pass


class AuthenticationError(RCMSException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(RCMSException):
    """Raised when authorization fails."""
    pass


class APIError(RCMSException):
    """Raised when API operations fail."""
    pass


class ExternalServiceError(RCMSException):
    """Raised when external service calls fail."""
    pass


class PreprocessingError(RCMSException):
    """Raised when preprocessing operations fail."""
    pass


class OCRError(PreprocessingError):
    """Raised when OCR processing fails."""
    pass


class ASRError(PreprocessingError):
    """Raised when speech recognition fails."""
    pass


class ComputerVisionError(RCMSException):
    """Raised when computer vision operations fail."""
    pass


class SentimentAnalysisError(RCMSException):
    """Raised when sentiment analysis fails."""
    pass


class PipelineError(RCMSException):
    """Raised when ML pipeline operations fail."""
    pass