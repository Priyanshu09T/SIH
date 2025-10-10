"""
Utility functions for RCMS application.
Common helper functions used across modules.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Make python-magic optional
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Generate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal hash string
    """
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def get_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect file type using python-magic if available, fallback to mimetypes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    try:
        if MAGIC_AVAILABLE:
            import magic
            mime = magic.Magic(mime=True)
            return mime.from_file(str(file_path))
        else:
            # Fallback to mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type or "application/octet-stream"
    except Exception:
        # Final fallback to mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"


def is_image_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is an image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is an image, False otherwise
    """
    mime_type = get_file_type(file_path)
    return mime_type.startswith('image/')


def is_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is an audio file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is an audio file, False otherwise
    """
    mime_type = get_file_type(file_path)
    return mime_type.startswith('audio/')


def is_video_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is a video file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a video file, False otherwise
    """
    mime_type = get_file_type(file_path)
    return mime_type.startswith('video/')


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(text: str) -> str:
    """
    Clean and normalize text input.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters (keep basic punctuation)
    # This is a basic implementation - can be enhanced
    import re
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate file extension against allowed list.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (e.g., ['.jpg', '.png'])
        
    Returns:
        True if extension is allowed, False otherwise
    """
    file_extension = Path(filename).suffix.lower()
    return file_extension in [ext.lower() for ext in allowed_extensions]


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate unique filename with timestamp prefix.
    
    Args:
        original_filename: Original filename
        
    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_part = Path(original_filename).stem
    extension = Path(original_filename).suffix
    
    return f"{timestamp}_{name_part}{extension}"


def safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with nested key support.
    
    Args:
        data: Dictionary to search
        key: Key (supports dot notation like 'models.category.threshold')
        default: Default value if key not found
        
    Returns:
        Value from dictionary or default
    """
    try:
        keys = key.split('.')
        value = data
        
        for k in keys:
            value = value[k]
        
        return value
    except (KeyError, TypeError):
        return default


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes into human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix