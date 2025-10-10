"""
Priority Classification Model for RCMS.
Determines urgency level of railway complaints.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from src.models.base_model import SklearnModel, ModelPrediction
from src.preprocessing.text_processor import TextProcessor
from src.utils.logger import LoggerMixin
from src.utils.exceptions import ModelError


class PriorityClassifier(SklearnModel):
    """
    Railway complaint priority classification model.
    
    Priority Levels:
    - Critical: Immediate safety threats, emergencies
    - High: Serious issues affecting service
    - Medium: Standard complaints needing attention
    - Low: Minor issues, suggestions
    """
    
    PRIORITIES = ["Critical", "High", "Medium", "Low"]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize priority classifier.
        
        Args:
            model_path: Path to saved model file
        """
        super().__init__(
            model_name="priority_classifier",
            model_class=Pipeline,
            model_params={},
            model_path=model_path
        )
        
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self.classes = self.PRIORITIES
    
    def _generate_sample_data(self) -> tuple:
        """
        Generate sample training data with priority labels.
        
        Returns:
            Tuple of (texts, priority_labels)
        """
        sample_data = [
            # Critical Priority
            ("Fire in the train coach! Emergency help needed!", "Critical"),
            ("Train accident occurred, people injured", "Critical"),
            ("Bomb threat received at station", "Critical"),
            ("Medical emergency passenger unconscious", "Critical"),
            ("Train derailed near bridge", "Critical"),
            ("Gas leak detected in pantry car", "Critical"),
            ("Electric shock from train door", "Critical"),
            ("Serious fight between passengers with weapons", "Critical"),
            
            # High Priority
            ("Train door not closing, safety issue", "High"),
            ("Theft of passenger luggage reported", "High"),
            ("Harassment of female passenger", "High"),
            ("Brake failure in train, emergency", "High"),
            ("No water in train for 12 hours", "High"),
            ("Food poisoning from train catering", "High"),
            ("Overcrowding causing safety concerns", "High"),
            ("Signal failure causing train delays", "High"),
            ("Platform collapse risk observed", "High"),
            ("AC failure in summer, passengers suffocating", "High"),
            
            # Medium Priority
            ("Toilet is dirty and needs cleaning", "Medium"),
            ("Train running 2 hours late", "Medium"),
            ("Food quality is poor in pantry car", "Medium"),
            ("Conductor was rude to passengers", "Medium"),
            ("Seat reservation not available", "Medium"),
            ("Announcement system not working", "Medium"),
            ("Charging points not functioning", "Medium"),
            ("Coach cleaning not done properly", "Medium"),
            ("Ticket checker asked for extra money", "Medium"),
            ("Platform lights not working at night", "Medium"),
            
            # Low Priority
            ("Suggestion to improve food menu", "Low"),
            ("Request for more trains on this route", "Low"),
            ("Feedback about staff uniforms", "Low"),
            ("Minor stain on seat cover", "Low"),
            ("Suggestion for digital displays", "Low"),
            ("Request for more vendors on platform", "Low"),
            ("Feedback about train timing", "Low"),
            ("Small scratch on window", "Low"),
            ("Suggestion for better announcements", "Low"),
            ("Request for more comfortable seats", "Low"),
        ]
        
        texts, labels = zip(*sample_data)
        return list(texts), list(labels)
    
    def _extract_priority_features(self, text: str) -> Dict[str, float]:
        """
        Extract priority-specific features from text.
        
        Args:
            text: Complaint text
            
        Returns:
            Dictionary of priority features
        """
        text_lower = text.lower()
        
        # Emergency keywords
        emergency_words = [
            'emergency', 'urgent', 'immediate', 'critical', 'help', 'fire', 
            'accident', 'injury', 'blood', 'unconscious', 'medical', 'bomb',
            'threat', 'danger', 'death', 'dying', 'panic', 'terror'
        ]
        
        # High priority indicators
        high_priority_words = [
            'theft', 'robbery', 'harassment', 'assault', 'violence', 'fight',
            'broken', 'failure', 'not working', 'derailed', 'collision',
            'poisoning', 'sick', 'vomiting', 'fever', 'serious', 'major'
        ]
        
        # Medium priority indicators  
        medium_priority_words = [
            'dirty', 'unclean', 'delayed', 'late', 'rude', 'poor', 'bad',
            'complaint', 'issue', 'problem', 'concern', 'unsatisfied'
        ]
        
        # Low priority indicators
        low_priority_words = [
            'suggestion', 'feedback', 'improve', 'better', 'request',
            'minor', 'small', 'little', 'slight', 'recommend'
        ]
        
        features = {}
        
        # Count emergency words
        emergency_count = sum(1 for word in emergency_words if word in text_lower)
        features['emergency_score'] = emergency_count / len(emergency_words)
        
        # Count high priority words
        high_count = sum(1 for word in high_priority_words if word in text_lower)
        features['high_priority_score'] = high_count / len(high_priority_words)
        
        # Count medium priority words
        medium_count = sum(1 for word in medium_priority_words if word in text_lower)
        features['medium_priority_score'] = medium_count / len(medium_priority_words)
        
        # Count low priority words
        low_count = sum(1 for word in low_priority_words if word in text_lower)
        features['low_priority_score'] = low_count / len(low_priority_words)
        
        # Text characteristics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def train(self, texts: Optional[List[str]] = None, labels: Optional[List[str]] = None, **kwargs) -> None:
        """
        Train the priority classification model.
        
        Args:
            texts: List of complaint texts
            labels: List of corresponding priorities
            **kwargs: Additional training parameters
        """
        if texts is None or labels is None:
            self.logger.info("No training data provided, using sample data for demonstration")
            texts, labels = self._generate_sample_data()
        
        try:
            self.logger.info(f"Training priority classifier with {len(texts)} samples...")
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                processed = self.text_processor.process(text, extract_features=False)
                processed_texts.append(processed.cleaned)
            
            # Train the pipeline
            self.model.fit(processed_texts, labels)
            self.is_trained = True
            self.classes = self.PRIORITIES
            
            # Calculate training accuracy
            train_pred = self.model.predict(processed_texts)
            accuracy = accuracy_score(labels, train_pred)
            
            self.logger.info(f"Priority classifier trained successfully. Accuracy: {accuracy:.3f}")
            
            # Print classification report
            report = classification_report(labels, train_pred, target_names=self.PRIORITIES)
            self.logger.info(f"Training Classification Report:\n{report}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise ModelError(f"Priority classifier training failed: {e}")
    
    def predict(self, text: str) -> ModelPrediction:
        """
        Predict priority for a complaint text.
        
        Args:
            text: Complaint text
            
        Returns:
            ModelPrediction with priority and confidence
        """
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            # Preprocess text
            processed = self.text_processor.process(text, extract_features=True)
            
            # Get urgency from text processor
            urgency_info = processed.metadata.get('urgency', {})
            urgency_level = urgency_info.get('level', 'low')
            urgency_score = urgency_info.get('urgency_score', 0)
            
            # Get ML prediction
            ml_prediction = self.model.predict([processed.cleaned])[0]
            
            # Get probabilities
            probabilities = None
            confidence = 1.0
            
            if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
                proba = self.model.predict_proba([processed.cleaned])[0]
                confidence = float(proba.max())
                probabilities = dict(zip(self.PRIORITIES, proba.tolist()))
            
            # Combine ML prediction with rule-based urgency
            final_prediction = self._combine_predictions(ml_prediction, urgency_level, urgency_score)
            
            return ModelPrediction(
                prediction=final_prediction,
                confidence=confidence,
                probabilities=probabilities,
                metadata={
                    'model_name': self.model_name,
                    'ml_prediction': ml_prediction,
                    'urgency_level': urgency_level,
                    'urgency_score': urgency_score,
                    'processed_text': processed.cleaned,
                    'original_text': text
                }
            )
            
        except Exception as e:
            self.logger.error(f"Priority prediction failed: {e}")
            raise ModelError(f"Priority prediction failed: {e}")
    
    def _combine_predictions(self, ml_prediction: str, urgency_level: str, urgency_score: int) -> str:
        """
        Combine ML prediction with rule-based urgency detection.
        
        Args:
            ml_prediction: Prediction from ML model
            urgency_level: Level from text processor
            urgency_score: Score from text processor
            
        Returns:
            Final priority prediction
        """
        # Map urgency levels to priorities
        urgency_to_priority = {
            'critical': 'Critical',
            'high': 'High', 
            'medium': 'Medium',
            'low': 'Low'
        }
        
        rule_based_priority = urgency_to_priority.get(urgency_level, 'Low')
        
        # Priority mapping for comparison
        priority_levels = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        
        ml_level = priority_levels.get(ml_prediction, 1)
        rule_level = priority_levels.get(rule_based_priority, 1)
        
        # Take the higher priority (more urgent)
        final_level = max(ml_level, rule_level)
        
        # Map back to priority name
        for priority, level in priority_levels.items():
            if level == final_level:
                return priority
        
        return 'Medium'  # Default fallback
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get priority probabilities for a complaint text.
        
        Args:
            text: Complaint text
            
        Returns:
            Dictionary of priority probabilities
        """
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            processed = self.text_processor.process(text, extract_features=False)
            proba = self.model.predict_proba([processed.cleaned])[0]
            return dict(zip(self.PRIORITIES, proba.tolist()))
            
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise ModelError(f"Probability prediction failed: {e}")
    
    def batch_predict(self, texts: List[str]) -> List[ModelPrediction]:
        """
        Predict priorities for multiple texts.
        
        Args:
            texts: List of complaint texts
            
        Returns:
            List of ModelPrediction objects
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to predict priority for text: {text[:50]}... Error: {e}")
                results.append(ModelPrediction(
                    prediction="Medium",
                    confidence=0.0,
                    metadata={'error': str(e)}
                ))
        
        return results


def create_priority_classifier(train_immediately: bool = True) -> PriorityClassifier:
    """
    Factory function to create and optionally train a priority classifier.
    
    Args:
        train_immediately: Whether to train with sample data immediately
        
    Returns:
        PriorityClassifier instance
    """
    classifier = PriorityClassifier()
    
    if train_immediately:
        classifier.train()
    
    return classifier