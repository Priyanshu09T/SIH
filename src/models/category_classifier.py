"""
Category Classification Model for RCMS.
Classifies railway complaints into predefined categories.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.models.base_model import SklearnModel, ModelPrediction
from src.preprocessing.text_processor import TextProcessor
from src.utils.logger import LoggerMixin
from src.utils.exceptions import ModelError


class CategoryClassifier(SklearnModel):
    """
    Railway complaint category classification model.
    
    Categories:
    - Infrastructure: Train, platform, facilities issues
    - Cleanliness: Hygiene, sanitation problems  
    - Safety: Security, safety concerns
    - Staff: Personnel behavior, service quality
    - Food: Catering, food quality issues
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
        """
        Initialize category classifier.
        
        Args:
            model_path: Path to saved model file
        """
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Create pipeline with steps
        pipeline_steps = [
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ]
        
        # Use RandomForest for robust classification
        super().__init__(
            model_name="category_classifier",
            model_class=Pipeline,
            model_params={'steps': pipeline_steps},
            model_path=model_path
        )
            min_samples_split=5
        )
        
        # Create pipeline
        self.model = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self.classes = self.CATEGORIES
    
    def _generate_sample_data(self) -> tuple:
        """
        Generate sample training data for demonstration.
        In production, this would load from a real dataset.
        
        Returns:
            Tuple of (texts, labels)
        """
        sample_data = [
            # Infrastructure
            ("The train seat is broken and uncomfortable", "Infrastructure"),
            ("Platform has cracks and needs repair", "Infrastructure"),
            ("Air conditioning not working in coach", "Infrastructure"),
            ("Train door is not closing properly", "Infrastructure"),
            ("Windows are dirty and cannot be opened", "Infrastructure"),
            ("Coach lighting is very dim", "Infrastructure"),
            ("Station building needs maintenance", "Infrastructure"),
            ("Track condition is poor with vibrations", "Infrastructure"),
            
            # Cleanliness
            ("Toilet is very dirty and smells bad", "Cleanliness"),
            ("Garbage everywhere in the compartment", "Cleanliness"),
            ("Coach floor is not cleaned properly", "Cleanliness"),
            ("Washroom has no water and soap", "Cleanliness"),
            ("Platform is littered with waste", "Cleanliness"),
            ("Train seats are stained and unclean", "Cleanliness"),
            ("Drinking water tastes bad", "Cleanliness"),
            ("Bad odor in the entire coach", "Cleanliness"),
            
            # Safety
            ("Suspicious person in the compartment", "Safety"),
            ("No security guard on the platform", "Safety"),
            ("Emergency chain not working", "Safety"),
            ("Fire extinguisher is missing", "Safety"),
            ("Platform gap is too wide and dangerous", "Safety"),
            ("No safety announcements made", "Safety"),
            ("Overcrowding causing safety issues", "Safety"),
            ("Electrical wires are exposed", "Safety"),
            
            # Staff
            ("Conductor was very rude to passengers", "Staff"),
            ("TTE asked for bribe", "Staff"),
            ("Station master was unhelpful", "Staff"),
            ("Cleaning staff not doing their job", "Staff"),
            ("Guard was sleeping during journey", "Staff"),
            ("Catering staff behaved badly", "Staff"),
            ("Ticket checker was unprofessional", "Staff"),
            ("Staff not wearing proper uniform", "Staff"),
            
            # Food
            ("Food quality is very poor", "Food"),
            ("Overpriced meals in pantry car", "Food"),
            ("Food served was stale", "Food"),
            ("No vegetarian options available", "Food"),
            ("Tea was cold and tasteless", "Food"),
            ("Snacks were expired", "Food"),
            ("Catering service is slow", "Food"),
            ("No food available on long journey", "Food"),
            
            # Other
            ("Ticket booking website is not working", "Other"),
            ("Wrong information given about timings", "Other"),
            ("Refund process is too complicated", "Other"),
            ("Mobile charging points not working", "Other"),
            ("Announcement system is unclear", "Other"),
            ("Lost luggage not traced properly", "Other"),
            ("Reservation chart not displayed", "Other"),
            ("General inquiry not handled well", "Other"),
        ]
        
        texts, labels = zip(*sample_data)
        return list(texts), list(labels)
    
    def train(self, texts: Optional[List[str]] = None, labels: Optional[List[str]] = None, **kwargs) -> None:
        """
        Train the category classification model.
        
        Args:
            texts: List of complaint texts
            labels: List of corresponding categories
            **kwargs: Additional training parameters
        """
        if texts is None or labels is None:
            self.logger.info("No training data provided, using sample data for demonstration")
            texts, labels = self._generate_sample_data()
        
        try:
            self.logger.info(f"Training category classifier with {len(texts)} samples...")
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                processed = self.text_processor.process(text, extract_features=False)
                processed_texts.append(processed.cleaned)
            
            # Train the pipeline
            self.model.fit(processed_texts, labels)
            self.is_trained = True
            self.classes = self.CATEGORIES
            
            # Calculate training accuracy
            train_pred = self.model.predict(processed_texts)
            accuracy = accuracy_score(labels, train_pred)
            
            self.logger.info(f"Category classifier trained successfully. Accuracy: {accuracy:.3f}")
            
            # Print classification report
            report = classification_report(labels, train_pred, target_names=self.CATEGORIES)
            self.logger.info(f"Training Classification Report:\n{report}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise ModelError(f"Category classifier training failed: {e}")
    
    def predict(self, text: str) -> ModelPrediction:
        """
        Predict category for a complaint text.
        
        Args:
            text: Complaint text
            
        Returns:
            ModelPrediction with category and confidence
        """
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            # Preprocess text
            processed = self.text_processor.process(text, extract_features=False)
            
            # Get prediction
            prediction = self.model.predict([processed.cleaned])[0]
            
            # Get probabilities
            probabilities = None
            confidence = 1.0
            
            if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
                proba = self.model.predict_proba([processed.cleaned])[0]
                confidence = float(proba.max())
                probabilities = dict(zip(self.CATEGORIES, proba.tolist()))
            
            return ModelPrediction(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                metadata={
                    'model_name': self.model_name,
                    'processed_text': processed.cleaned,
                    'original_text': text
                }
            )
            
        except Exception as e:
            self.logger.error(f"Category prediction failed: {e}")
            raise ModelError(f"Category prediction failed: {e}")
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get category probabilities for a complaint text.
        
        Args:
            text: Complaint text
            
        Returns:
            Dictionary of category probabilities
        """
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            processed = self.text_processor.process(text, extract_features=False)
            proba = self.model.predict_proba([processed.cleaned])[0]
            return dict(zip(self.CATEGORIES, proba.tolist()))
            
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise ModelError(f"Probability prediction failed: {e}")
    
    def batch_predict(self, texts: List[str]) -> List[ModelPrediction]:
        """
        Predict categories for multiple texts.
        
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
                self.logger.warning(f"Failed to predict for text: {text[:50]}... Error: {e}")
                results.append(ModelPrediction(
                    prediction="Other",
                    confidence=0.0,
                    metadata={'error': str(e)}
                ))
        
        return results
    
    def get_category_keywords(self) -> Dict[str, List[str]]:
        """
        Get important keywords for each category.
        
        Returns:
            Dictionary of category keywords
        """
        if not self.is_trained:
            return {}
        
        try:
            # Get feature names from vectorizer
            feature_names = self.model.named_steps['vectorizer'].get_feature_names_out()
            
            # Get feature importance from classifier
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                importances = self.model.named_steps['classifier'].feature_importances_
                
                # Get top features
                top_indices = np.argsort(importances)[-50:]  # Top 50 features
                top_features = [feature_names[i] for i in top_indices]
                
                return {'top_keywords': top_features}
            
        except Exception as e:
            self.logger.warning(f"Could not extract keywords: {e}")
        
        return {}


def create_category_classifier(train_immediately: bool = True) -> CategoryClassifier:
    """
    Factory function to create and optionally train a category classifier.
    
    Args:
        train_immediately: Whether to train with sample data immediately
        
    Returns:
        CategoryClassifier instance
    """
    classifier = CategoryClassifier()
    
    if train_immediately:
        classifier.train()
    
    return classifier