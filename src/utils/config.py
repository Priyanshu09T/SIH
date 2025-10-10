"""
Configuration management for RCMS.
Handles loading and validation of application settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Application
    app_name: str = "Railway Complaint Management System"
    app_version: str = "1.0.0"
    debug: bool = True
    environment: str = "development"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Database
    database_url: str = "postgresql+asyncpg://rcms_user:rcms_pass@localhost:5432/rcms_db"
    database_echo: bool = False
    
    # Security
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Features
    enable_sentiment: bool = True
    enable_computer_vision: bool = True
    enable_speech_recognition: bool = True
    
    # File Upload
    max_file_size_mb: int = 50
    upload_path: str = "data/media"
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        # For development, allow default key with warning
        if v == "your-secret-key-here":
            import warnings
            warnings.warn("Using default secret key - change this in production!")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "settings.yaml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model-specific configuration.
    
    Args:
        model_name: Name of the model (category, priority, escalation, sentiment)
        
    Returns:
        Model configuration dictionary
    """
    config = load_config()
    models_config = config.get('models', {})
    
    if model_name not in models_config:
        raise ValueError(f"Model configuration not found: {model_name}")
    
    return models_config[model_name]


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a feature is enabled in configuration.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        True if feature is enabled, False otherwise
    """
    config = load_config()
    features = config.get('features', {})
    return features.get(feature_name, False)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings