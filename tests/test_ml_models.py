"""
Comprehensive tests for ML models in RCMS.
Tests all classification models and the complete pipeline.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.models.category_classifier import CategoryClassifier, create_category_classifier
from src.models.priority_classifier import PriorityClassifier, create_priority_classifier
from src.models.escalation_classifier import EscalationClassifier, create_escalation_classifier
from src.models.ml_pipeline import ComplaintMLPipeline, create_pipeline, ComplaintAnalysis
from src.utils.exceptions import ModelError, PipelineError


class TestCategoryClassifier:
    """Test cases for CategoryClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = CategoryClassifier()
        assert classifier.model_name == "category_classifier"
        assert not classifier.is_trained
        assert classifier.classes == CategoryClassifier.COMPLAINT_CATEGORIES
    
    def test_training_with_sample_data(self):
        """Test training with generated sample data."""
        classifier = CategoryClassifier()
        classifier.train()
        
        assert classifier.is_trained
        assert classifier.model is not None
    
    def test_prediction_before_training(self):
        """Test that prediction fails before training."""
        classifier = CategoryClassifier()
        
        with pytest.raises(ModelError, match="Model is not trained"):
            classifier.predict("Train is late")
    
    def test_prediction_after_training(self):
        """Test prediction functionality after training."""
        classifier = create_category_classifier(train_immediately=True)
        
        # Test infrastructure complaint
        result = classifier.predict("Platform roof is leaking badly")
        assert result.prediction in CategoryClassifier.COMPLAINT_CATEGORIES
        assert 0 <= result.confidence <= 1
        assert result.probabilities is not None
        assert len(result.probabilities) == len(CategoryClassifier.COMPLAINT_CATEGORIES)
        
        # Test safety complaint
        result = classifier.predict("Emergency brakes not working properly")
        assert result.prediction in CategoryClassifier.COMPLAINT_CATEGORIES
        
        # Test staff complaint
        result = classifier.predict("Conductor was very rude to passengers")
        assert result.prediction in CategoryClassifier.COMPLAINT_CATEGORIES
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        classifier = create_category_classifier(train_immediately=True)
        
        texts = [
            "Toilet is very dirty",
            "Train accident happened",
            "Food was spoiled"
        ]
        
        results = classifier.predict_batch(texts)
        assert len(results) == len(texts)
        
        for result in results:
            assert result.prediction in CategoryClassifier.COMPLAINT_CATEGORIES
            assert 0 <= result.confidence <= 1
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            
            # Train and save model
            classifier1 = create_category_classifier(train_immediately=True)
            classifier1.save(model_path)
            
            # Load model in new instance
            classifier2 = CategoryClassifier(model_path)
            classifier2.load(model_path)
            
            assert classifier2.is_trained
            
            # Test that both models give same prediction
            test_text = "Train is running very late"
            result1 = classifier1.predict(test_text)
            result2 = classifier2.predict(test_text)
            
            assert result1.prediction == result2.prediction


class TestPriorityClassifier:
    """Test cases for PriorityClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = PriorityClassifier()
        assert classifier.model_name == "priority_classifier"
        assert not classifier.is_trained
        assert classifier.classes == PriorityClassifier.PRIORITY_LEVELS
    
    def test_training_with_sample_data(self):
        """Test training with generated sample data."""
        classifier = PriorityClassifier()
        classifier.train()
        
        assert classifier.is_trained
        assert classifier.model is not None
    
    def test_prediction_with_urgency_detection(self):
        """Test that urgency detection influences priority."""
        classifier = create_priority_classifier(train_immediately=True)
        
        # Test critical urgency words
        result = classifier.predict("URGENT: Train accident with injuries")
        assert result.prediction in PriorityClassifier.PRIORITY_LEVELS
        assert result.metadata['urgency_detected'] is True
        
        # Test non-urgent complaint
        result = classifier.predict("Suggestion to improve food quality")
        assert result.prediction in PriorityClassifier.PRIORITY_LEVELS
    
    def test_priority_rules_application(self):
        """Test that business rules are applied correctly."""
        classifier = create_priority_classifier(train_immediately=True)
        
        # Test emergency keywords trigger high priority
        result = classifier.predict("Emergency: Fire in train compartment")
        assert result.prediction in ['Critical', 'High']
        
        # Test safety keywords
        result = classifier.predict("Safety issue: Broken emergency chain")
        assert result.prediction in ['Critical', 'High']


class TestEscalationClassifier:
    """Test cases for EscalationClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = EscalationClassifier()
        assert classifier.model_name == "escalation_classifier"
        assert not classifier.is_trained
        assert classifier.classes == EscalationClassifier.ESCALATION_CLASSES
    
    def test_training_with_sample_data(self):
        """Test training with generated sample data."""
        classifier = EscalationClassifier()
        classifier.train()
        
        assert classifier.is_trained
        assert classifier.model is not None
    
    def test_escalation_rules(self):
        """Test that escalation business rules work correctly."""
        classifier = create_escalation_classifier(train_immediately=True)
        
        # Test auto-escalation for corruption
        result = classifier.predict("Staff demanding bribes from passengers", 
                                  category="Staff", priority="High")
        assert result.prediction == "Escalate"
        
        # Test auto-escalation for critical priority
        result = classifier.predict("Train derailment", 
                                  category="Safety", priority="Critical")
        assert result.prediction == "Escalate"
        
        # Test simple boolean method
        should_escalate = classifier.should_escalate("Minor toilet cleaning issue", 
                                                   category="Cleanliness", priority="Low")
        assert isinstance(should_escalate, bool)
    
    def test_escalation_features_extraction(self):
        """Test escalation-specific feature extraction."""
        classifier = EscalationClassifier()
        
        features = classifier._extract_escalation_features(
            "Repeated corruption by staff, no action taken",
            category="Staff",
            priority="High"
        )
        
        assert isinstance(features, dict)
        assert 'escalation_keywords' in features
        assert 'severity_keywords' in features
        assert 'frequency_keywords' in features
        assert features['escalation_keywords'] > 0  # Should detect "corruption"


class TestComplaintMLPipeline:
    """Test cases for the complete ML pipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = ComplaintMLPipeline()
        assert not pipeline.is_initialized
        assert pipeline.category_classifier is not None
        assert pipeline.priority_classifier is not None
        assert pipeline.escalation_classifier is not None
    
    def test_pipeline_initialization_with_training(self):
        """Test pipeline initialization with model training."""
        pipeline = ComplaintMLPipeline()
        pipeline.initialize(train_models=True)
        
        assert pipeline.is_initialized
        assert pipeline.category_classifier.is_trained
        assert pipeline.priority_classifier.is_trained
        assert pipeline.escalation_classifier.is_trained
    
    def test_complaint_analysis_before_initialization(self):
        """Test that analysis fails before initialization."""
        pipeline = ComplaintMLPipeline()
        
        with pytest.raises(PipelineError, match="Pipeline not initialized"):
            pipeline.analyze_complaint("Test complaint")
    
    def test_complete_complaint_analysis(self):
        """Test complete end-to-end complaint analysis."""
        pipeline = create_pipeline(train_immediately=True)
        
        complaint_text = "URGENT: Train accident near station, multiple injuries reported"
        
        analysis = pipeline.analyze_complaint(complaint_text, "TEST_001")
        
        # Verify analysis structure
        assert isinstance(analysis, ComplaintAnalysis)
        assert analysis.complaint_id == "TEST_001"
        assert analysis.text == complaint_text
        assert analysis.category in CategoryClassifier.COMPLAINT_CATEGORIES
        assert analysis.priority in PriorityClassifier.PRIORITY_LEVELS
        assert analysis.escalation in EscalationClassifier.ESCALATION_CLASSES
        
        # Verify confidence scores
        assert 0 <= analysis.category_confidence <= 1
        assert 0 <= analysis.priority_confidence <= 1
        assert 0 <= analysis.escalation_confidence <= 1
        
        # Verify processing metadata
        assert analysis.processing_time_ms > 0
        assert len(analysis.processed_text) > 0
        assert isinstance(analysis.keywords_extracted, list)
        assert isinstance(analysis.urgency_detected, bool)
        
        # Verify recommendations
        assert len(analysis.recommendation) > 0
        assert len(analysis.action_required) > 0
        
        # Test JSON serialization
        json_str = analysis.to_json()
        assert isinstance(json_str, str)
        assert "complaint_id" in json_str
    
    def test_batch_analysis(self):
        """Test batch complaint analysis."""
        pipeline = create_pipeline(train_immediately=True)
        
        complaints = [
            ("BATCH_001", "Platform is very dirty and needs cleaning"),
            ("BATCH_002", "Emergency: Fire in train compartment B2"),
            ("BATCH_003", "Food quality in pantry car is poor")
        ]
        
        results = pipeline.analyze_batch(complaints)
        
        assert len(results) == len(complaints)
        
        for i, analysis in enumerate(results):
            assert analysis.complaint_id == complaints[i][0]
            assert analysis.text == complaints[i][1]
            assert analysis.category in CategoryClassifier.COMPLAINT_CATEGORIES
    
    def test_model_info_retrieval(self):
        """Test model information retrieval."""
        pipeline = create_pipeline(train_immediately=True)
        
        info = pipeline.get_model_info()
        
        assert info['pipeline_initialized'] is True
        assert 'models' in info
        assert 'category_classifier' in info['models']
        assert 'priority_classifier' in info['models']
        assert 'escalation_classifier' in info['models']
        assert 'text_processor' in info
        
        # Check that all models are reported as trained
        for model_name in ['category_classifier', 'priority_classifier', 'escalation_classifier']:
            assert info['models'][model_name]['trained'] is True
    
    def test_save_and_load_models(self):
        """Test saving and loading all models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and train pipeline
            pipeline1 = create_pipeline(train_immediately=True)
            
            # Save models
            pipeline1.save_models(temp_dir)
            
            # Verify files exist
            assert os.path.exists(os.path.join(temp_dir, "category_model.pkl"))
            assert os.path.exists(os.path.join(temp_dir, "priority_model.pkl"))
            assert os.path.exists(os.path.join(temp_dir, "escalation_model.pkl"))
            
            # Create new pipeline and load models
            pipeline2 = ComplaintMLPipeline()
            pipeline2.load_models(temp_dir)
            
            assert pipeline2.is_initialized
            
            # Test that both pipelines give same results
            test_text = "Train is running very late today"
            
            analysis1 = pipeline1.analyze_complaint(test_text, "LOAD_TEST_1")
            analysis2 = pipeline2.analyze_complaint(test_text, "LOAD_TEST_2")
            
            assert analysis1.category == analysis2.category
            assert analysis1.priority == analysis2.priority
            assert analysis1.escalation == analysis2.escalation
    
    def test_recommendation_generation(self):
        """Test recommendation generation logic."""
        pipeline = create_pipeline(train_immediately=True)
        
        # Test critical escalation case
        analysis = pipeline.analyze_complaint(
            "URGENT: Staff demanding bribes from passengers",
            "REC_TEST_001"
        )
        
        assert "ESCALATION" in analysis.recommendation.upper()
        assert len(analysis.action_required) > 0
        
        # Test routine case
        analysis = pipeline.analyze_complaint(
            "Suggestion to improve food variety in pantry car",
            "REC_TEST_002"
        )
        
        assert analysis.recommendation is not None
        assert analysis.action_required is not None
    
    @patch('src.models.category_classifier.CategoryClassifier.predict')
    def test_pipeline_error_handling(self, mock_predict):
        """Test pipeline error handling when models fail."""
        # Mock a model failure
        mock_predict.side_effect = Exception("Model prediction failed")
        
        pipeline = create_pipeline(train_immediately=True)
        
        with pytest.raises(PipelineError, match="Complaint analysis failed"):
            pipeline.analyze_complaint("Test complaint")


def test_factory_functions():
    """Test all factory functions work correctly."""
    # Test category classifier factory
    category_classifier = create_category_classifier(train_immediately=True)
    assert category_classifier.is_trained
    
    # Test priority classifier factory
    priority_classifier = create_priority_classifier(train_immediately=True)
    assert priority_classifier.is_trained
    
    # Test escalation classifier factory
    escalation_classifier = create_escalation_classifier(train_immediately=True)
    assert escalation_classifier.is_trained
    
    # Test pipeline factory
    pipeline = create_pipeline(train_immediately=True)
    assert pipeline.is_initialized


def test_integration_scenario():
    """Test a complete realistic scenario."""
    # Create the complete pipeline
    pipeline = create_pipeline(train_immediately=True)
    
    # Test various types of complaints
    test_complaints = [
        ("INTEGRATION_001", "Emergency brake failure on Express train, passengers at risk"),
        ("INTEGRATION_002", "Platform toilet needs basic cleaning"),
        ("INTEGRATION_003", "Repeated corruption by TTE, no action despite multiple complaints"),
        ("INTEGRATION_004", "Food poisoning from railway catering, multiple passengers affected"),
        ("INTEGRATION_005", "Suggestion to add more charging points in coaches")
    ]
    
    # Analyze all complaints
    results = pipeline.analyze_batch(test_complaints)
    
    assert len(results) == 5
    
    # Verify first complaint (emergency) gets high priority and escalation
    emergency_analysis = results[0]
    assert emergency_analysis.priority in ['Critical', 'High']
    assert emergency_analysis.escalation == "Escalate"
    
    # Verify routine complaint gets appropriate handling
    routine_analysis = results[1]
    assert routine_analysis.priority in ['Low', 'Medium']
    
    # Verify corruption complaint gets escalated
    corruption_analysis = results[2]
    assert corruption_analysis.escalation == "Escalate"
    
    # Verify all analyses have proper structure
    for analysis in results:
        assert analysis.category in CategoryClassifier.COMPLAINT_CATEGORIES
        assert analysis.priority in PriorityClassifier.PRIORITY_LEVELS
        assert analysis.escalation in EscalationClassifier.ESCALATION_CLASSES
        assert analysis.processing_time_ms > 0
        assert len(analysis.recommendation) > 0


if __name__ == "__main__":
    # Run a simple test
    print("Testing ML Models...")
    
    # Test individual models
    print("✓ Testing Category Classifier...")
    category_classifier = create_category_classifier(train_immediately=True)
    result = category_classifier.predict("Train is running very late")
    print(f"  Category: {result.prediction} (confidence: {result.confidence:.3f})")
    
    print("✓ Testing Priority Classifier...")
    priority_classifier = create_priority_classifier(train_immediately=True)
    result = priority_classifier.predict("URGENT: Emergency brake failure")
    print(f"  Priority: {result.prediction} (confidence: {result.confidence:.3f})")
    
    print("✓ Testing Escalation Classifier...")
    escalation_classifier = create_escalation_classifier(train_immediately=True)
    result = escalation_classifier.predict("Staff demanding bribes")
    print(f"  Escalation: {result.prediction} (confidence: {result.confidence:.3f})")
    
    print("✓ Testing Complete Pipeline...")
    pipeline = create_pipeline(train_immediately=True)
    analysis = pipeline.analyze_complaint("URGENT: Train accident with injuries")
    print(f"  Complete Analysis:")
    print(f"    Category: {analysis.category}")
    print(f"    Priority: {analysis.priority}")
    print(f"    Escalation: {analysis.escalation}")
    print(f"    Processing Time: {analysis.processing_time_ms:.2f}ms")
    
    print("\n🎉 All ML model tests passed!")