"""
Base ML Model Interface for RCMS.
Provides common functionality for all ML models.
"""

import os
import pickle
import joblib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from src.utils.logger import LoggerMixin
from src.utils.exceptions import ModelError, ModelNotFoundError, ModelLoadError, PredictionError


@dataclass
class ModelPrediction:
    """Container for model prediction results."""
    prediction: Union[str, int, float, List]
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModel(ABC, LoggerMixin):
    """
    Abstract base class for all ML models in RCMS.
    
    Provides common functionality:
    - Model loading/saving
    - Prediction interface
    - Error handling
    - Logging
    """

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            model_path: Path to saved model file
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.classes = []
        
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> ModelPrediction:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction object
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> Dict[str, float]:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of class probabilities
        """
        pass
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            path: Optional path to save model
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ModelError("Cannot save untrained model")
        
        save_path = path or self.model_path
        if not save_path:
            save_path = f"models/{self.model_name}.joblib"
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save using joblib for scikit-learn compatibility
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'classes': self.classes,
                'model_name': self.model_name,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, save_path)
            self.model_path = save_path
            
            self.logger.info(f"Model saved successfully to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Model save failed: {e}")
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Optional path to load model from
        """
        load_path = path or self.model_path
        if not load_path:
            raise ModelError("No model path specified")
        
        if not os.path.exists(load_path):
            raise ModelNotFoundError(f"Model file not found: {load_path}")
        
        try:
            model_data = joblib.load(load_path)
            
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', [])
            self.classes = model_data.get('classes', [])
            self.model_name = model_data.get('model_name', self.model_name)
            self.is_trained = model_data.get('is_trained', True)
            self.model_path = load_path
            
            self.logger.info(f"Model loaded successfully from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model load failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'class_count': len(self.classes),
            'classes': self.classes,
            'model_type': self.__class__.__name__
        }
    
    def validate_input(self, X: Any) -> None:
        """
        Validate input data for prediction.
        
        Args:
            X: Input data to validate
        """
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        if self.model is None:
            raise ModelError("Model is not loaded")
    
    def _ensure_model_dir(self) -> None:
        """Ensure models directory exists."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)


class SklearnModel(BaseModel):
    """
    Base class for scikit-learn based models.
    Provides common sklearn functionality.
    """
    
    def __init__(self, model_name: str, model_class, model_params: Optional[Dict] = None, model_path: Optional[str] = None):
        """
        Initialize sklearn model.
        
        Args:
            model_name: Name of the model
            model_class: Scikit-learn model class
            model_params: Model parameters
            model_path: Path to saved model file
        """
        super().__init__(model_name, model_path)
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = model_class(**self.model_params)
    
    def train(self, X: Any, y: Any, **kwargs) -> None:
        """
        Train the sklearn model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional parameters
        """
        try:
            self.logger.info(f"Training {self.model_name} model...")
            
            self.model.fit(X, y)
            self.is_trained = True
            
            # Store feature names if available
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1] if hasattr(X, 'shape') else len(X[0]))]
            
            # Store classes
            if hasattr(self.model, 'classes_'):
                self.classes = list(self.model.classes_)
            
            self.logger.info(f"Model {self.model_name} trained successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed for {self.model_name}: {e}")
            raise ModelError(f"Training failed: {e}")
    
    def predict(self, X: Any) -> ModelPrediction:
        """
        Make prediction using sklearn model.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction object
        """
        self.validate_input(X)
        
        try:
            prediction = self.model.predict(X)
            
            # Get probabilities if available
            probabilities = None
            confidence = 1.0
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
                if len(proba.shape) > 1:
                    # Multi-class case
                    confidence = float(proba.max())
                    probabilities = dict(zip(self.classes, proba[0].tolist()))
                else:
                    # Binary case
                    confidence = float(max(proba[0], 1 - proba[0]))
            
            # Handle single prediction vs batch
            if hasattr(prediction, '__len__') and len(prediction) == 1:
                prediction = prediction[0]
            
            return ModelPrediction(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                metadata={
                    'model_name': self.model_name,
                    'feature_count': len(self.feature_names)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {self.model_name}: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def predict_proba(self, X: Any) -> Dict[str, float]:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of class probabilities
        """
        self.validate_input(X)
        
        if not hasattr(self.model, 'predict_proba'):
            raise ModelError(f"Model {self.model_name} does not support probability prediction")
        
        try:
            proba = self.model.predict_proba(X)
            if len(proba.shape) > 1:
                return dict(zip(self.classes, proba[0].tolist()))
            else:
                return {'positive': float(proba[0]), 'negative': float(1 - proba[0])}
                
        except Exception as e:
            self.logger.error(f"Probability prediction failed for {self.model_name}: {e}")
            raise PredictionError(f"Probability prediction failed: {e}")