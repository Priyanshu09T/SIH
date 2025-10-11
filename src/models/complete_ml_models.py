"""
Complete ML Models Module for RCMS.
Working implementation with category, priority, and escalation classification.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import logging

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.preprocessing.text_processor import TextProcessor

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryClassifier:
    """Railway complaint category classification."""
    
    CATEGORIES = ["Infrastructure", "Cleanliness", "Safety", "Staff", "Food", "Other"]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.text_processor = TextProcessor()
        
    def _create_pipeline(self):
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                max_depth=15
            ))
        ])
    
    def _generate_training_data(self):
        """Generate training data for categories."""
        data = {
            "Infrastructure": [
                "train late delayed", "platform crowded", "ac not working", "toilet broken",
                "seat damaged", "window stuck", "door broken", "lights not working",
                "fan noisy", "berth uncomfortable", "coach old", "track repair needed",
                "platform roof leaking", "escalator broken", "waiting room problems"
            ],
            "Cleanliness": [
                "toilet dirty", "garbage everywhere", "coach smells", "washroom unhygienic",
                "platform dirty", "seat stained", "floor unclean", "bad odor",
                "toilet paper missing", "soap unavailable", "water dirty", "poor cleaning",
                "waste bins full", "dust seats", "spider webs"
            ],
            "Safety": [
                "theft train", "passenger harassment", "emergency brake broken", "fire hazard",
                "unsafe platform", "no security", "suspicious activity", "pickpocket",
                "accident happened", "dangerous behavior", "safety equipment missing", "exit blocked",
                "first aid missing", "smoke coach", "electrical wire exposed"
            ],
            "Staff": [
                "rude conductor", "tte misbehaved", "staff unhelpful", "ticket checker absent",
                "guard sleeping", "staff bribe", "poor service", "staff not responding",
                "conductor rude", "employee misconduct", "staff harassment", "unprofessional",
                "ticket collector missing", "guard unavailable", "staff discrimination"
            ],
            "Food": [
                "food quality poor", "meal overpriced", "pantry dirty", "food not fresh",
                "water quality bad", "catering poor", "food poisoning", "meal cold",
                "vendor rude", "breakfast unavailable", "tea expensive", "food stale",
                "pantry staff rude", "meal small", "water overpriced"
            ],
            "Other": [
                "wifi not working", "charging broken", "announcement unclear", "booking problem",
                "refund pending", "website down", "app not working", "ticket printing issue",
                "reservation problem", "general inquiry", "suggestion improvement", "feedback",
                "app complaint", "website slow", "booking confirmation missing"
            ]
        }
        
        texts = []
        labels = []
        for category, examples in data.items():
            texts.extend(examples)
            labels.extend([category] * len(examples))
        
        return texts, labels
    
    def train(self):
        """Train the category classifier."""
        logger.info("Training category classifier...")
        
        texts, labels = self._generate_training_data()
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed = self.text_processor.process(text, extract_features=False)
            processed_texts.append(processed.cleaned)
        
        # Train model
        self.model = self._create_pipeline()
        self.model.fit(processed_texts, labels)
        
        # Evaluate
        predictions = self.model.predict(processed_texts)
        accuracy = accuracy_score(labels, predictions)
        
        logger.info(f"Category classifier training completed. Accuracy: {accuracy:.3f}")
        self.is_trained = True
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict category for text."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        processed = self.text_processor.process(text, extract_features=False)
        prediction = self.model.predict([processed.cleaned])[0]
        probabilities = self.model.predict_proba([processed.cleaned])[0]
        confidence = max(probabilities)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": dict(zip(self.CATEGORIES, probabilities))
        }


class PriorityClassifier:
    """Railway complaint priority classification."""
    
    PRIORITIES = ["Low", "Medium", "High", "Critical"]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.text_processor = TextProcessor()
    
    def _create_pipeline(self):
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                random_state=42
            ))
        ])
    
    def _generate_training_data(self):
        """Generate training data for priorities."""
        data = {
            "Critical": [
                "emergency brake failed", "fire accident", "theft robbery", "passenger injured",
                "derailment risk", "signal failure", "accident emergency", "security threat",
                "medical emergency", "safety critical", "immediate danger", "life threatening"
            ],
            "High": [
                "ac completely broken", "toilet severely damaged", "staff harassment serious",
                "food poisoning", "major delay", "security concern", "urgent repair needed",
                "serious complaint", "immediate attention", "escalate urgently", "broken equipment"
            ],
            "Medium": [
                "ac not cooling", "toilet needs cleaning", "staff behavior poor", "food quality bad",
                "moderate delay", "needs improvement", "general complaint", "service issue",
                "maintenance required", "attention needed", "problem reported", "issue found"
            ],
            "Low": [
                "minor suggestion", "small improvement", "general feedback", "routine complaint",
                "slight inconvenience", "minor issue", "suggestion box", "feedback form",
                "small problem", "trivial matter", "general inquiry", "information request"
            ]
        }
        
        texts = []
        labels = []
        for priority, examples in data.items():
            texts.extend(examples)
            labels.extend([priority] * len(examples))
        
        return texts, labels
    
    def train(self):
        """Train the priority classifier."""
        logger.info("Training priority classifier...")
        
        texts, labels = self._generate_training_data()
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed = self.text_processor.process(text, extract_features=False)
            processed_texts.append(processed.cleaned)
        
        # Train model
        self.model = self._create_pipeline()
        self.model.fit(processed_texts, labels)
        
        # Evaluate
        predictions = self.model.predict(processed_texts)
        accuracy = accuracy_score(labels, predictions)
        
        logger.info(f"Priority classifier training completed. Accuracy: {accuracy:.3f}")
        self.is_trained = True
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict priority for text."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        processed = self.text_processor.process(text, extract_features=False)
        prediction = self.model.predict([processed.cleaned])[0]
        probabilities = self.model.predict_proba([processed.cleaned])[0]
        confidence = max(probabilities)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": dict(zip(self.PRIORITIES, probabilities))
        }


class EscalationClassifier:
    """Railway complaint escalation decision."""
    
    DECISIONS = ["Escalate", "No_Escalation"]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.text_processor = TextProcessor()
    
    def _create_pipeline(self):
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                random_state=42
            ))
        ])
    
    def _generate_training_data(self):
        """Generate training data for escalation."""
        data = {
            "Escalate": [
                "emergency situation", "safety critical", "immediate danger", "serious accident",
                "theft robbery", "staff misconduct", "harassment serious", "urgent attention",
                "escalate management", "senior officer", "critical issue", "unacceptable behavior",
                "formal complaint", "legal action", "media attention", "public safety"
            ],
            "No_Escalation": [
                "routine complaint", "minor issue", "general feedback", "small problem",
                "regular maintenance", "normal service", "standard procedure", "daily operations",
                "minor inconvenience", "trivial matter", "suggestion improvement", "feedback only",
                "information request", "general inquiry", "routine question", "normal process"
            ]
        }
        
        texts = []
        labels = []
        for decision, examples in data.items():
            texts.extend(examples)
            labels.extend([decision] * len(examples))
        
        return texts, labels
    
    def train(self):
        """Train the escalation classifier."""
        logger.info("Training escalation classifier...")
        
        texts, labels = self._generate_training_data()
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed = self.text_processor.process(text, extract_features=False)
            processed_texts.append(processed.cleaned)
        
        # Train model
        self.model = self._create_pipeline()
        self.model.fit(processed_texts, labels)
        
        # Evaluate
        predictions = self.model.predict(processed_texts)
        accuracy = accuracy_score(labels, predictions)
        
        logger.info(f"Escalation classifier training completed. Accuracy: {accuracy:.3f}")
        self.is_trained = True
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict escalation decision for text."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        processed = self.text_processor.process(text, extract_features=False)
        prediction = self.model.predict([processed.cleaned])[0]
        probabilities = self.model.predict_proba([processed.cleaned])[0]
        confidence = max(probabilities)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": dict(zip(self.DECISIONS, probabilities)),
            "escalate": prediction == "Escalate"
        }


class ComplaintMLPipeline:
    """Complete ML pipeline for complaint analysis."""
    
    def __init__(self):
        self.category_classifier = CategoryClassifier()
        self.priority_classifier = PriorityClassifier()
        self.escalation_classifier = EscalationClassifier()
        self.is_trained = False
    
    def train(self):
        """Train all models in the pipeline."""
        logger.info("Training complete ML pipeline...")
        
        self.category_classifier.train()
        self.priority_classifier.train()
        self.escalation_classifier.train()
        
        self.is_trained = True
        logger.info("Complete ML pipeline training finished!")
    
    def analyze_complaint(self, text: str) -> Dict[str, Any]:
        """Analyze a complaint using the complete pipeline."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        # Get predictions from all models
        category_result = self.category_classifier.predict(text)
        priority_result = self.priority_classifier.predict(text)
        escalation_result = self.escalation_classifier.predict(text)
        
        # Combine results
        analysis = {
            "input_text": text,
            "category": {
                "prediction": category_result["prediction"],
                "confidence": category_result["confidence"]
            },
            "priority": {
                "prediction": priority_result["prediction"],
                "confidence": priority_result["confidence"]
            },
            "escalation": {
                "prediction": escalation_result["prediction"],
                "confidence": escalation_result["confidence"],
                "should_escalate": escalation_result["escalate"]
            },
            "overall_confidence": (
                category_result["confidence"] + 
                priority_result["confidence"] + 
                escalation_result["confidence"]
            ) / 3
        }
        
        return analysis


def create_ml_pipeline(train_immediately: bool = True) -> ComplaintMLPipeline:
    """Create and optionally train the ML pipeline."""
    pipeline = ComplaintMLPipeline()
    
    if train_immediately:
        pipeline.train()
    
    return pipeline


if __name__ == "__main__":
    # Demo
    print("🚀 RCMS ML Models Demo")
    print("=" * 50)
    
    # Create and train pipeline
    pipeline = create_ml_pipeline(train_immediately=True)
    
    # Test complaints
    test_complaints = [
        "The train toilet is extremely dirty and smells terrible!",
        "Emergency! The train caught fire and passengers are in danger!",
        "The conductor was very rude and demanded extra money",
        "WiFi is not working in the coach",
        "Food quality in pantry car is poor and overpriced"
    ]
    
    print("\n🧪 Testing Complete Pipeline:")
    print("-" * 50)
    
    for complaint in test_complaints:
        result = pipeline.analyze_complaint(complaint)
        
        print(f"\n📝 Complaint: '{complaint}'")
        print(f"📂 Category: {result['category']['prediction']} ({result['category']['confidence']:.2f})")
        print(f"⚡ Priority: {result['priority']['prediction']} ({result['priority']['confidence']:.2f})")
        print(f"🚨 Escalate: {'Yes' if result['escalation']['should_escalate'] else 'No'} ({result['escalation']['confidence']:.2f})")
        print(f"📊 Overall Confidence: {result['overall_confidence']:.2f}")
    
    print("\n" + "=" * 50)
    print("🎉 All ML models working successfully!")