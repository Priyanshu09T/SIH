"""
Escalation Decision Model for RCMS.
Determines whether a complaint should be escalated to higher authorities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from src.models.base_model import SklearnModel, ModelPrediction
from src.preprocessing.text_processor import TextProcessor
from src.utils.logger import LoggerMixin
from src.utils.exceptions import ModelError


class EscalationClassifier(SklearnModel):
    """
    Railway complaint escalation classification model.
    
    Escalation Decisions:
    - True: Escalate to higher authorities/management
    - False: Handle at current level
    """
    
    ESCALATION_CLASSES = ["Escalate", "No_Escalation"]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize escalation classifier.
        
        Args:
            model_path: Path to saved model file
        """
        super().__init__(
            model_name="escalation_classifier",
            model_class=Pipeline,
            model_params={},
            model_path=model_path
        )
        
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.model = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self.classes = self.ESCALATION_CLASSES
    
    def _generate_sample_data(self) -> tuple:
        """
        Generate sample training data with escalation labels.
        
        Returns:
            Tuple of (texts, escalation_labels)
        """
        sample_data = [
            # Escalate - Serious safety issues
            ("Train accident caused multiple injuries", "Escalate"),
            ("Staff taking bribes from passengers", "Escalate"),
            ("Sexual harassment by train staff", "Escalate"),
            ("Food poisoning from railway catering", "Escalate"),
            ("Bomb threat at railway station", "Escalate"),
            ("Fire in train compartment", "Escalate"),
            ("Major structural damage to platform", "Escalate"),
            ("Staff physically assaulting passengers", "Escalate"),
            ("Systematic corruption in ticket booking", "Escalate"),
            ("Medical negligence in railway hospital", "Escalate"),
            
            # Escalate - Repeated/unresolved issues
            ("Complained 5 times about same issue, no action", "Escalate"),
            ("Station master ignoring passenger complaints", "Escalate"),
            ("Chronic water shortage in trains for months", "Escalate"),
            ("Repeated theft incidents with no security action", "Escalate"),
            ("Train consistently late by 4+ hours daily", "Escalate"),
            ("Platform roof collapsed, injuring passengers", "Escalate"),
            ("TTE demanding illegal payments regularly", "Escalate"),
            ("Discrimination against disabled passengers", "Escalate"),
            ("Station facilities closed for weeks without notice", "Escalate"),
            ("Emergency services not available during crisis", "Escalate"),
            
            # No Escalation - Routine issues
            ("Toilet needs cleaning in coach B2", "No_Escalation"),
            ("Food quality could be better", "No_Escalation"),
            ("Seat reservation chart not displayed clearly", "No_Escalation"),
            ("Train running 30 minutes late", "No_Escalation"),
            ("Conductor was a bit rude", "No_Escalation"),
            ("Coach lighting is dim", "No_Escalation"),
            ("Platform needs better cleaning", "No_Escalation"),
            ("Suggestion to improve announcements", "No_Escalation"),
            ("Charging point not working in coach", "No_Escalation"),
            ("Request for more vendors on platform", "No_Escalation"),
            
            # No Escalation - Minor service issues
            ("Tea was cold in pantry car", "No_Escalation"),
            ("Seat cover has small stain", "No_Escalation"),
            ("Window blind not working properly", "No_Escalation"),
            ("Magazine not available in coach", "No_Escalation"),
            ("Air conditioning could be cooler", "No_Escalation"),
            ("Platform bench needs repair", "No_Escalation"),
            ("Drinking water tap has low pressure", "No_Escalation"),
            ("Train timing could be more convenient", "No_Escalation"),
            ("Feedback about staff uniform", "No_Escalation"),
            ("Request for more comfortable pillows", "No_Escalation"),
        ]
        
        texts, labels = zip(*sample_data)
        return list(texts), list(labels)
    
    def _extract_escalation_features(self, text: str, category: str = "", priority: str = "") -> Dict[str, float]:
        """
        Extract escalation-specific features from text and metadata.
        
        Args:
            text: Complaint text
            category: Predicted category
            priority: Predicted priority
            
        Returns:
            Dictionary of escalation features
        """
        text_lower = text.lower()
        
        # Escalation trigger words
        escalation_keywords = [
            'corruption', 'bribe', 'illegal', 'harassment', 'assault', 'violence',
            'discrimination', 'negligence', 'repeated', 'multiple times', 'no action',
            'ignore', 'systemic', 'chronic', 'escalate', 'management', 'senior',
            'accident', 'injury', 'danger', 'threat', 'emergency', 'crisis'
        ]
        
        # Severity indicators
        severity_keywords = [
            'serious', 'major', 'severe', 'critical', 'urgent', 'immediate',
            'life threatening', 'safety risk', 'public health', 'structural damage'
        ]
        
        # Complaint frequency indicators
        frequency_keywords = [
            'again', 'repeatedly', 'multiple times', 'every day', 'always',
            'continuously', 'persistent', 'ongoing', 'chronic'
        ]
        
        features = {}
        
        # Count escalation keywords
        escalation_count = sum(1 for word in escalation_keywords if word in text_lower)
        features['escalation_keywords'] = escalation_count
        
        # Count severity keywords
        severity_count = sum(1 for word in severity_keywords if word in text_lower)
        features['severity_keywords'] = severity_count
        
        # Count frequency keywords
        frequency_count = sum(1 for word in frequency_keywords if word in text_lower)
        features['frequency_keywords'] = frequency_count
        
        # Category-based features
        high_escalation_categories = ['Safety', 'Staff']
        features['high_escalation_category'] = 1 if category in high_escalation_categories else 0
        
        # Priority-based features
        features['critical_priority'] = 1 if priority == 'Critical' else 0
        features['high_priority'] = 1 if priority in ['Critical', 'High'] else 0
        
        # Text characteristics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def train(self, texts: Optional[List[str]] = None, labels: Optional[List[str]] = None, **kwargs) -> None:
        """
        Train the escalation classification model.
        
        Args:
            texts: List of complaint texts
            labels: List of corresponding escalation decisions
            **kwargs: Additional training parameters
        """
        if texts is None or labels is None:
            self.logger.info("No training data provided, using sample data for demonstration")
            texts, labels = self._generate_sample_data()
        
        try:
            self.logger.info(f"Training escalation classifier with {len(texts)} samples...")
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                processed = self.text_processor.process(text, extract_features=False)
                processed_texts.append(processed.cleaned)
            
            # Train the pipeline
            self.model.fit(processed_texts, labels)
            self.is_trained = True
            self.classes = self.ESCALATION_CLASSES
            
            # Calculate training accuracy
            train_pred = self.model.predict(processed_texts)
            accuracy = accuracy_score(labels, train_pred)
            
            self.logger.info(f"Escalation classifier trained successfully. Accuracy: {accuracy:.3f}")
            
            # Print classification report
            report = classification_report(labels, train_pred, target_names=self.ESCALATION_CLASSES)
            self.logger.info(f"Training Classification Report:\n{report}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise ModelError(f"Escalation classifier training failed: {e}")
    
    def predict(self, text: str, category: str = "", priority: str = "") -> ModelPrediction:
        """
        Predict escalation decision for a complaint.
        
        Args:
            text: Complaint text
            category: Predicted category (optional)
            priority: Predicted priority (optional)
            
        Returns:
            ModelPrediction with escalation decision
        """
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            # Preprocess text
            processed = self.text_processor.process(text, extract_features=True)
            
            # Get ML prediction
            ml_prediction = self.model.predict([processed.cleaned])[0]
            
            # Get probabilities
            probabilities = None
            confidence = 1.0
            
            if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
                proba = self.model.predict_proba([processed.cleaned])[0]
                confidence = float(proba.max())
                probabilities = dict(zip(self.ESCALATION_CLASSES, proba.tolist()))
            
            # Apply business rules
            final_prediction = self._apply_escalation_rules(
                ml_prediction, text, category, priority, processed.metadata
            )
            
            return ModelPrediction(
                prediction=final_prediction,
                confidence=confidence,
                probabilities=probabilities,
                metadata={
                    'model_name': self.model_name,
                    'ml_prediction': ml_prediction,
                    'category': category,
                    'priority': priority,
                    'processed_text': processed.cleaned,
                    'original_text': text,
                    'escalation_features': self._extract_escalation_features(text, category, priority)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Escalation prediction failed: {e}")
            raise ModelError(f"Escalation prediction failed: {e}")
    
    def _apply_escalation_rules(self, ml_prediction: str, text: str, category: str, 
                              priority: str, metadata: Dict) -> str:
        """
        Apply business rules to refine escalation decision.
        
        Args:
            ml_prediction: Prediction from ML model
            text: Original complaint text
            category: Complaint category
            priority: Complaint priority
            metadata: Additional metadata
            
        Returns:
            Final escalation decision
        """
        # Automatic escalation triggers
        auto_escalate_conditions = [
            priority == 'Critical',
            'bribe' in text.lower() or 'corruption' in text.lower(),
            'harassment' in text.lower() or 'assault' in text.lower(),
            'accident' in text.lower() and 'injury' in text.lower(),
            'repeated' in text.lower() and 'no action' in text.lower()
        ]
        
        # If any auto-escalation condition is met, escalate
        if any(auto_escalate_conditions):
            return "Escalate"
        
        # Automatic no-escalation triggers
        auto_no_escalate_conditions = [
            priority == 'Low',
            'suggestion' in text.lower() or 'feedback' in text.lower(),
            category == 'Other' and priority in ['Low', 'Medium']
        ]
        
        # If any no-escalation condition is met and no escalation triggers, don't escalate
        if any(auto_no_escalate_conditions) and not any(auto_escalate_conditions):
            return "No_Escalation"
        
        # Otherwise, use ML prediction
        return ml_prediction
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get escalation probabilities for a complaint text.
        
        Args:
            text: Complaint text
            
        Returns:
            Dictionary of escalation probabilities
        """
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            processed = self.text_processor.process(text, extract_features=False)
            proba = self.model.predict_proba([processed.cleaned])[0]
            return dict(zip(self.ESCALATION_CLASSES, proba.tolist()))
            
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise ModelError(f"Probability prediction failed: {e}")
    
    def should_escalate(self, text: str, category: str = "", priority: str = "") -> bool:
        """
        Simple boolean escalation decision.
        
        Args:
            text: Complaint text
            category: Predicted category (optional)
            priority: Predicted priority (optional)
            
        Returns:
            True if should escalate, False otherwise
        """
        result = self.predict(text, category, priority)
        return result.prediction == "Escalate"


def create_escalation_classifier(train_immediately: bool = True) -> EscalationClassifier:
    """
    Factory function to create and optionally train an escalation classifier.
    
    Args:
        train_immediately: Whether to train with sample data immediately
        
    Returns:
        EscalationClassifier instance
    """
    classifier = EscalationClassifier()
    
    if train_immediately:
        classifier.train()
    
    return classifier