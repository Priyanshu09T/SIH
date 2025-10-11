"""
Simplified Category Classification Model for RCMS.
Railway complaint category classification using scikit-learn.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src.preprocessing.text_processor import TextProcessor
from src.utils.logger import LoggerMixin
from src.utils.exceptions import ModelError


class CategoryClassifier(LoggerMixin):
    """
    Railway complaint category classification model.
    
    Categories:
    - Infrastructure: Train, platform, facilities issues
    - Cleanliness: Hygiene, sanitation problems
    - Safety: Security, emergency, danger issues
    - Staff: Employee behavior, service quality
    - Food: Catering, meal quality issues
    - Other: Miscellaneous complaints
    """
    
    CATEGORIES = [
        "Infrastructure",
        "Cleanliness", 
        "Safety",
        "Staff",
        "Food",
        "Other"
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize category classifier."""
        self.model_path = model_path
        self.text_processor = TextProcessor()
        self.model = None
        self.is_trained = False
        
    def _create_pipeline(self) -> Pipeline:
        """Create the ML pipeline."""
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
    
    def _generate_training_data(self) -> Tuple[List[str], List[str]]:
        """Generate synthetic training data for the classifier."""
        
        # Infrastructure complaints
        infrastructure_texts = [
            "train is late", "platform is crowded", "ac not working", "toilet broken",
            "seat is damaged", "window won't close", "door stuck", "lights not working",
            "fan making noise", "berth is uncomfortable", "coach is old", "railway track needs repair",
            "platform roof leaking", "escalator not working", "waiting room dirty"
        ]
        
        # Cleanliness complaints  
        cleanliness_texts = [
            "toilet is very dirty", "garbage everywhere", "coach smells bad", "washroom unhygienic",
            "platform dirty", "seat stained", "floor not clean", "bad odor in coach",
            "toilet paper missing", "soap not available", "water dirty", "cleaning poor",
            "waste bins overflowing", "spider webs in corner", "dust on seats"
        ]
        
        # Safety complaints
        safety_texts = [
            "theft in train", "harassment by passenger", "emergency brake not working", "fire hazard",
            "unsafe platform", "no security guard", "suspicious activity", "pickpocket warning",
            "accident happened", "dangerous behavior", "safety equipment missing", "emergency exit blocked",
            "first aid kit empty", "smoke in coach", "electrical wire exposed"
        ]
        
        # Staff complaints
        staff_texts = [
            "rude conductor", "tte misbehaved", "staff unhelpful", "ticket checker absent",
            "guard sleeping", "staff demanding bribe", "poor service", "staff not responding",
            "conductor rude behavior", "employee misconduct", "staff harassment", "unprofessional behavior",
            "ticket collector missing", "guard not available", "staff discrimination"
        ]
        
        # Food complaints
        food_texts = [
            "food quality poor", "meal overpriced", "pantry car dirty", "food not fresh",
            "water quality bad", "catering service poor", "food poisoning", "meal cold",
            "vendor rude", "breakfast not available", "tea too expensive", "food stale",
            "pantry staff rude", "meal portion small", "water bottle overpriced"
        ]
        
        # Other complaints
        other_texts = [
            "wifi not working", "charging point broken", "announcement unclear", "booking problem",
            "refund pending", "website down", "app not working", "ticket printing issue",
            "reservation problem", "general inquiry", "suggestion for improvement", "feedback",
            "complaint about app", "website slow", "booking confirmation missing"
        ]
        
        # Combine all data
        texts = (infrastructure_texts + cleanliness_texts + safety_texts + 
                staff_texts + food_texts + other_texts)
        
        labels = (["Infrastructure"] * len(infrastructure_texts) +
                 ["Cleanliness"] * len(cleanliness_texts) +
                 ["Safety"] * len(safety_texts) +
                 ["Staff"] * len(staff_texts) +
                 ["Food"] * len(food_texts) +
                 ["Other"] * len(other_texts))
        
        return texts, labels
    
    def train(self, texts: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> None:
        """Train the category classifier."""
        try:
            # Use provided data or generate synthetic data
            if texts is None or labels is None:
                self.logger.info("No training data provided, generating synthetic data...")
                texts, labels = self._generate_training_data()
            
            self.logger.info(f"Training category classifier with {len(texts)} samples...")
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                processed = self.text_processor.process(text, extract_features=False)
                processed_texts.append(processed.cleaned)
            
            # Create and train pipeline
            self.model = self._create_pipeline()
            self.model.fit(processed_texts, labels)
            
            # Evaluate on training data
            train_pred = self.model.predict(processed_texts)
            accuracy = accuracy_score(labels, train_pred)
            
            self.logger.info(f"Training completed. Accuracy: {accuracy:.3f}")
            
            # Print classification report
            report = classification_report(labels, train_pred, target_names=self.CATEGORIES)
            self.logger.info(f"Classification Report:\\n{report}")
            
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise ModelError(f"Category classifier training failed: {e}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict category for a given complaint text."""
        if not self.is_trained:
            raise ModelError("Model is not trained. Call train() first.")
        
        try:
            # Preprocess text
            processed = self.text_processor.process(text, extract_features=False)
            
            # Make prediction
            prediction = self.model.predict([processed.cleaned])[0]
            probabilities = self.model.predict_proba([processed.cleaned])[0]
            confidence = max(probabilities)
            
            # Get all class probabilities
            class_probabilities = dict(zip(self.CATEGORIES, probabilities))
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "probabilities": class_probabilities,
                "processed_text": processed.cleaned
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Category prediction failed: {e}")
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict categories for multiple texts."""
        if not self.is_trained:
            raise ModelError("Model is not trained. Call train() first.")
        
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to predict for text: {text[:50]}... Error: {e}")
                results.append({
                    "prediction": "Other",
                    "confidence": 0.0,
                    "probabilities": {},
                    "error": str(e)
                })
        
        return results


def create_category_classifier(train_immediately: bool = False) -> CategoryClassifier:
    """
    Factory function to create and optionally train a category classifier.
    
    Args:
        train_immediately: Whether to train the model immediately
        
    Returns:
        CategoryClassifier instance
    """
    classifier = CategoryClassifier()
    
    if train_immediately:
        classifier.train()
    
    return classifier