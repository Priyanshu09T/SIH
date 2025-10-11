"""
Enhanced ML Models for RCMS with Improved Feature Engineering
Better category distinction, refined escalation logic, and comprehensive training data
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import logging

# Add src to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.preprocessing.enhanced_text_processor import EnhancedTextProcessor
from src.models.enhanced_training_data import EnhancedTrainingDataGenerator

logger = logging.getLogger(__name__)


class EnhancedCategoryClassifier:
    """Enhanced category classifier with better feature engineering."""
    
    CATEGORIES = ["Infrastructure", "Cleanliness", "Safety", "Staff", "Food", "Other"]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.text_processor = EnhancedTextProcessor()
        self.training_data_generator = EnhancedTrainingDataGenerator()
        
    def _create_pipeline(self):
        """Create enhanced pipeline with better parameters."""
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,  # Increased features
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams
                min_df=2,  # Require at least 2 occurrences
                max_df=0.9,  # Ignore too common words
                sublinear_tf=True,  # Use sublinear scaling
                analyzer='word'
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,  # More trees
                random_state=42,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'  # Handle class imbalance
            ))
        ])
    
    def _extract_enhanced_features(self, text: str) -> str:
        """Extract enhanced features and create feature-rich text."""
        features = self.text_processor.process(text, extract_features=True)
        
        # Create enhanced text with feature indicators
        enhanced_text = features.cleaned
        
        # Add category indicator features
        if features.infrastructure_indicators:
            enhanced_text += " INFRASTRUCTURE_CATEGORY " + " ".join(features.infrastructure_indicators)
        
        if features.safety_indicators:
            enhanced_text += " SAFETY_CATEGORY " + " ".join(features.safety_indicators)
        
        if features.cleanliness_indicators:
            enhanced_text += " CLEANLINESS_CATEGORY " + " ".join(features.cleanliness_indicators)
        
        if features.staff_indicators:
            enhanced_text += " STAFF_CATEGORY " + " ".join(features.staff_indicators)
        
        if features.food_indicators:
            enhanced_text += " FOOD_CATEGORY " + " ".join(features.food_indicators)
        
        # Add railway-specific features
        if features.railway_keywords:
            enhanced_text += " RAILWAY_TERMS " + " ".join(features.railway_keywords)
        
        # Add urgency indicators
        if features.urgency_score > 0.5:
            enhanced_text += " HIGH_URGENCY"
        
        if features.has_emergency_context:
            enhanced_text += " EMERGENCY_CONTEXT"
        
        return enhanced_text
    
    def train(self):
        """Train enhanced category classifier."""
        logger.info("Training enhanced category classifier...")
        
        # Get enhanced training data
        texts, categories, _ = self.training_data_generator.get_balanced_training_data()
        
        # Process texts with enhanced features
        enhanced_texts = []
        for text in texts:
            enhanced_text = self._extract_enhanced_features(text)
            enhanced_texts.append(enhanced_text)
        
        # Train model
        self.model = self._create_pipeline()
        self.model.fit(enhanced_texts, categories)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.model, enhanced_texts, categories, cv=5, scoring='accuracy')
        avg_cv_score = np.mean(cv_scores)
        
        # Get training accuracy
        predictions = self.model.predict(enhanced_texts)
        training_accuracy = accuracy_score(categories, predictions)
        
        logger.info(f"Enhanced category classifier training completed.")
        logger.info(f"Training accuracy: {training_accuracy:.3f}")
        logger.info(f"Cross-validation accuracy: {avg_cv_score:.3f} (+/- {np.std(cv_scores) * 2:.3f})")
        
        self.is_trained = True
        return training_accuracy, avg_cv_score
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict category with enhanced features."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        enhanced_text = self._extract_enhanced_features(text)
        prediction = self.model.predict([enhanced_text])[0]
        probabilities = self.model.predict_proba([enhanced_text])[0]
        confidence = max(probabilities)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": dict(zip(self.CATEGORIES, probabilities))
        }


class EnhancedPriorityClassifier:
    """Enhanced priority classifier with urgency-aware features."""
    
    PRIORITIES = ["Low", "Medium", "High", "Critical"]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.text_processor = EnhancedTextProcessor()
        self.training_data_generator = EnhancedTrainingDataGenerator()
    
    def _create_pipeline(self):
        """Create enhanced pipeline for priority classification."""
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=80,
                random_state=42,
                max_depth=15,
                class_weight='balanced'
            ))
        ])
    
    def _extract_priority_features(self, text: str) -> str:
        """Extract priority-specific features."""
        features = self.text_processor.process(text, extract_features=True)
        
        enhanced_text = features.cleaned
        
        # Add urgency score as feature
        if features.urgency_score > 0.8:
            enhanced_text += " CRITICAL_URGENCY"
        elif features.urgency_score > 0.6:
            enhanced_text += " HIGH_URGENCY"
        elif features.urgency_score > 0.3:
            enhanced_text += " MEDIUM_URGENCY"
        
        # Add urgency indicators
        if features.urgency_indicators:
            enhanced_text += " URGENCY_WORDS " + " ".join(features.urgency_indicators)
        
        # Add emergency context
        if features.has_emergency_context:
            enhanced_text += " EMERGENCY_SITUATION"
        
        # Add safety indicators for high priority
        if features.safety_indicators:
            enhanced_text += " SAFETY_ISSUE " + " ".join(features.safety_indicators)
        
        return enhanced_text
    
    def train(self):
        """Train enhanced priority classifier."""
        logger.info("Training enhanced priority classifier...")
        
        # Get training data
        texts, _, priorities = self.training_data_generator.get_balanced_training_data()
        
        # Process texts with priority features
        enhanced_texts = []
        for text in texts:
            enhanced_text = self._extract_priority_features(text)
            enhanced_texts.append(enhanced_text)
        
        # Train model
        self.model = self._create_pipeline()
        self.model.fit(enhanced_texts, priorities)
        
        # Evaluate
        cv_scores = cross_val_score(self.model, enhanced_texts, priorities, cv=5, scoring='accuracy')
        avg_cv_score = np.mean(cv_scores)
        
        predictions = self.model.predict(enhanced_texts)
        training_accuracy = accuracy_score(priorities, predictions)
        
        logger.info(f"Enhanced priority classifier training completed.")
        logger.info(f"Training accuracy: {training_accuracy:.3f}")
        logger.info(f"Cross-validation accuracy: {avg_cv_score:.3f}")
        
        self.is_trained = True
        return training_accuracy, avg_cv_score
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict priority with enhanced features."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        enhanced_text = self._extract_priority_features(text)
        prediction = self.model.predict([enhanced_text])[0]
        probabilities = self.model.predict_proba([enhanced_text])[0]
        confidence = max(probabilities)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": dict(zip(self.PRIORITIES, probabilities))
        }


class EnhancedEscalationClassifier:
    """Enhanced escalation classifier with refined criteria."""
    
    DECISIONS = ["Escalate", "No_Escalation"]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.text_processor = EnhancedTextProcessor()
        self.training_data_generator = EnhancedTrainingDataGenerator()
    
    def _create_pipeline(self):
        """Create enhanced pipeline for escalation classification."""
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9,
                sublinear_tf=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=60,
                random_state=42,
                max_depth=12,
                class_weight={
                    'Escalate': 0.6,  # Slightly favor escalation for safety
                    'No_Escalation': 0.4
                }
            ))
        ])
    
    def _extract_escalation_features(self, text: str) -> str:
        """Extract escalation-specific features."""
        features = self.text_processor.process(text, extract_features=True)
        
        enhanced_text = features.cleaned
        
        # Add escalation keywords
        if features.escalation_keywords:
            enhanced_text += " ESCALATION_NEEDED " + " ".join(features.escalation_keywords)
        
        # Add high urgency features
        if features.urgency_score > 0.7:
            enhanced_text += " HIGH_URGENCY_ESCALATE"
        
        # Add emergency context
        if features.has_emergency_context:
            enhanced_text += " EMERGENCY_ESCALATE"
        
        # Add safety concerns
        if features.safety_indicators:
            enhanced_text += " SAFETY_ESCALATE"
        
        # Add negative sentiment for escalation consideration
        if len(features.negative_sentiment_words) > 2:
            enhanced_text += " NEGATIVE_SENTIMENT_HIGH"
        
        # Check for suggestion context (usually no escalation)
        if features.has_suggestion_context:
            enhanced_text += " SUGGESTION_NO_ESCALATE"
        
        return enhanced_text
    
    def train(self):
        """Train enhanced escalation classifier."""
        logger.info("Training enhanced escalation classifier...")
        
        # Get escalation training data
        texts, escalation_labels = self.training_data_generator.get_escalation_training_data()
        
        # Process texts with escalation features
        enhanced_texts = []
        for text in texts:
            enhanced_text = self._extract_escalation_features(text)
            enhanced_texts.append(enhanced_text)
        
        # Train model
        self.model = self._create_pipeline()
        self.model.fit(enhanced_texts, escalation_labels)
        
        # Evaluate
        cv_scores = cross_val_score(self.model, enhanced_texts, escalation_labels, cv=5, scoring='accuracy')
        avg_cv_score = np.mean(cv_scores)
        
        predictions = self.model.predict(enhanced_texts)
        training_accuracy = accuracy_score(escalation_labels, predictions)
        
        logger.info(f"Enhanced escalation classifier training completed.")
        logger.info(f"Training accuracy: {training_accuracy:.3f}")
        logger.info(f"Cross-validation accuracy: {avg_cv_score:.3f}")
        
        self.is_trained = True
        return training_accuracy, avg_cv_score
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict escalation with enhanced features."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        enhanced_text = self._extract_escalation_features(text)
        prediction = self.model.predict([enhanced_text])[0]
        probabilities = self.model.predict_proba([enhanced_text])[0]
        confidence = max(probabilities)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": dict(zip(self.DECISIONS, probabilities)),
            "escalate": prediction == "Escalate"
        }


class EnhancedComplaintMLPipeline:
    """Enhanced complete ML pipeline with improved models."""
    
    def __init__(self):
        self.category_classifier = EnhancedCategoryClassifier()
        self.priority_classifier = EnhancedPriorityClassifier()
        self.escalation_classifier = EnhancedEscalationClassifier()
        self.is_trained = False
        self.training_metrics = {}
    
    def train(self):
        """Train all enhanced models."""
        logger.info("Training enhanced ML pipeline...")
        
        # Train category classifier
        cat_train_acc, cat_cv_acc = self.category_classifier.train()
        
        # Train priority classifier
        pri_train_acc, pri_cv_acc = self.priority_classifier.train()
        
        # Train escalation classifier
        esc_train_acc, esc_cv_acc = self.escalation_classifier.train()
        
        # Store metrics
        self.training_metrics = {
            "category": {"training_accuracy": cat_train_acc, "cv_accuracy": cat_cv_acc},
            "priority": {"training_accuracy": pri_train_acc, "cv_accuracy": pri_cv_acc},
            "escalation": {"training_accuracy": esc_train_acc, "cv_accuracy": esc_cv_acc}
        }
        
        self.is_trained = True
        logger.info("Enhanced ML pipeline training completed!")
        
        return self.training_metrics
    
    def analyze_complaint(self, text: str) -> Dict[str, Any]:
        """Analyze complaint with enhanced pipeline."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        # Get predictions from all models
        category_result = self.category_classifier.predict(text)
        priority_result = self.priority_classifier.predict(text)
        escalation_result = self.escalation_classifier.predict(text)
        
        # Calculate overall confidence
        overall_confidence = (
            category_result["confidence"] * 0.4 +  # Category weight
            priority_result["confidence"] * 0.3 +   # Priority weight
            escalation_result["confidence"] * 0.3   # Escalation weight
        )
        
        return {
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
            "overall_confidence": overall_confidence,
            "enhanced_features": True
        }


def create_enhanced_ml_pipeline(train_immediately: bool = True) -> EnhancedComplaintMLPipeline:
    """Create and optionally train the enhanced ML pipeline."""
    pipeline = EnhancedComplaintMLPipeline()
    
    if train_immediately:
        metrics = pipeline.train()
        logger.info(f"Enhanced pipeline trained with metrics: {metrics}")
    
    return pipeline


if __name__ == "__main__":
    print("🚀 Enhanced RCMS ML Models Training")
    print("=" * 60)
    
    # Create and train enhanced pipeline
    pipeline = create_enhanced_ml_pipeline(train_immediately=True)
    
    # Print training metrics
    print(f"\n📊 Training Metrics:")
    for model_name, metrics in pipeline.training_metrics.items():
        print(f"  {model_name.title()}:")
        print(f"    Training Accuracy: {metrics['training_accuracy']:.3f}")
        print(f"    Cross-Validation: {metrics['cv_accuracy']:.3f}")
    
    # Test with problematic cases from unseen data
    print(f"\n🧪 Testing Enhanced Models on Previously Problematic Cases:")
    print("-" * 60)
    
    test_cases = [
        {
            "text": "The overhead wire is sparking dangerously near platform 3!",
            "expected": "Infrastructure/Critical/Escalate"
        },
        {
            "text": "Toilet is extremely dirty with cockroaches everywhere",
            "expected": "Cleanliness/High/Escalate"
        },
        {
            "text": "WiFi is not working properly in my coach",
            "expected": "Other/Low/No"
        },
        {
            "text": "Emergency! Train derailed and passengers injured",
            "expected": "Safety/Critical/Escalate"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = pipeline.analyze_complaint(case["text"])
        
        print(f"\n{i}. Text: '{case['text']}'")
        print(f"   Expected: {case['expected']}")
        print(f"   Predicted: {result['category']['prediction']}/{result['priority']['prediction']}/{'Escalate' if result['escalation']['should_escalate'] else 'No'}")
        print(f"   Confidence: Category={result['category']['confidence']:.2f}, Priority={result['priority']['confidence']:.2f}, Escalation={result['escalation']['confidence']:.2f}")
        print(f"   Overall: {result['overall_confidence']:.2f}")
    
    print(f"\n" + "=" * 60)
    print("🎉 Enhanced ML models training and testing complete!")