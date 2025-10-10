"""
ML Pipeline for Railway Complaint Management System.
Orchestrates all ML models for end-to-end complaint processing.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from src.models.category_classifier import CategoryClassifier
from src.models.priority_classifier import PriorityClassifier
from src.models.escalation_classifier import EscalationClassifier
from src.preprocessing.text_processor import TextProcessor
from src.utils.logger import LoggerMixin
from src.utils.exceptions import PipelineError


@dataclass
class ComplaintAnalysis:
    """
    Complete analysis result for a railway complaint.
    """
    # Original complaint data
    complaint_id: str
    text: str
    timestamp: datetime
    
    # ML predictions
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    escalation: str
    escalation_confidence: float
    
    # Processing metadata
    processing_time_ms: float
    processed_text: str
    keywords_extracted: List[str]
    urgency_detected: bool
    
    # Recommendation
    recommendation: str
    action_required: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ComplaintMLPipeline(LoggerMixin):
    """
    Complete ML pipeline for railway complaint processing.
    Integrates category, priority, and escalation classification.
    """
    
    def __init__(self, 
                 category_model_path: Optional[str] = None,
                 priority_model_path: Optional[str] = None,
                 escalation_model_path: Optional[str] = None):
        """
        Initialize the ML pipeline.
        
        Args:
            category_model_path: Path to saved category model
            priority_model_path: Path to saved priority model 
            escalation_model_path: Path to saved escalation model
        """
        super().__init__()
        
        # Initialize components
        self.text_processor = TextProcessor()
        
        # Initialize models
        self.category_classifier = CategoryClassifier(category_model_path)
        self.priority_classifier = PriorityClassifier(priority_model_path)
        self.escalation_classifier = EscalationClassifier(escalation_model_path)
        
        self.is_initialized = False
        
        self.logger.info("ML Pipeline initialized")
    
    def initialize(self, train_models: bool = True) -> None:
        """
        Initialize all models in the pipeline.
        
        Args:
            train_models: Whether to train models with sample data
        """
        try:
            self.logger.info("Initializing ML pipeline...")
            start_time = time.time()
            
            if train_models:
                # Train all models with sample data
                self.logger.info("Training category classifier...")
                self.category_classifier.train()
                
                self.logger.info("Training priority classifier...")
                self.priority_classifier.train()
                
                self.logger.info("Training escalation classifier...")
                self.escalation_classifier.train()
            
            self.is_initialized = True
            
            init_time = (time.time() - start_time) * 1000
            self.logger.info(f"ML Pipeline initialized successfully in {init_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            raise PipelineError(f"Pipeline initialization failed: {e}")
    
    def analyze_complaint(self, 
                         complaint_text: str, 
                         complaint_id: Optional[str] = None) -> ComplaintAnalysis:
        """
        Perform complete analysis of a railway complaint.
        
        Args:
            complaint_text: The complaint text to analyze
            complaint_id: Optional ID for the complaint
            
        Returns:
            ComplaintAnalysis with all predictions and metadata
        """
        if not self.is_initialized:
            raise PipelineError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            # Generate complaint ID if not provided
            if complaint_id is None:
                complaint_id = f"RCMS_{int(time.time())}"
            
            self.logger.info(f"Analyzing complaint {complaint_id}")
            
            # Step 1: Text preprocessing
            processed = self.text_processor.process(complaint_text, extract_features=True)
            
            # Step 2: Category classification
            category_result = self.category_classifier.predict(complaint_text)
            category = category_result.prediction
            category_confidence = category_result.confidence
            
            # Step 3: Priority classification
            priority_result = self.priority_classifier.predict(complaint_text)
            priority = priority_result.prediction
            priority_confidence = priority_result.confidence
            
            # Step 4: Escalation decision
            escalation_result = self.escalation_classifier.predict(
                complaint_text, category, priority
            )
            escalation = escalation_result.prediction
            escalation_confidence = escalation_result.confidence
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Generate recommendations
            recommendation, action_required = self._generate_recommendations(
                category, priority, escalation, processed.metadata
            )
            
            # Create analysis result
            analysis = ComplaintAnalysis(
                complaint_id=complaint_id,
                text=complaint_text,
                timestamp=datetime.now(),
                category=category,
                category_confidence=category_confidence,
                priority=priority,
                priority_confidence=priority_confidence,
                escalation=escalation,
                escalation_confidence=escalation_confidence,
                processing_time_ms=processing_time,
                processed_text=processed.cleaned,
                keywords_extracted=processed.keywords,
                urgency_detected=processed.metadata.get('urgency_detected', False),
                recommendation=recommendation,
                action_required=action_required
            )
            
            self.logger.info(
                f"Complaint analysis completed in {processing_time:.2f}ms. "
                f"Category: {category}, Priority: {priority}, Escalation: {escalation}"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Complaint analysis failed for {complaint_id}: {e}")
            raise PipelineError(f"Complaint analysis failed: {e}")
    
    def analyze_batch(self, complaints: List[Tuple[str, str]]) -> List[ComplaintAnalysis]:
        """
        Analyze multiple complaints in batch.
        
        Args:
            complaints: List of (complaint_id, complaint_text) tuples
            
        Returns:
            List of ComplaintAnalysis results
        """
        if not self.is_initialized:
            raise PipelineError("Pipeline not initialized. Call initialize() first.")
        
        self.logger.info(f"Starting batch analysis of {len(complaints)} complaints")
        start_time = time.time()
        
        results = []
        failed_complaints = []
        
        for complaint_id, complaint_text in complaints:
            try:
                analysis = self.analyze_complaint(complaint_text, complaint_id)
                results.append(analysis)
            except Exception as e:
                self.logger.error(f"Failed to analyze complaint {complaint_id}: {e}")
                failed_complaints.append(complaint_id)
        
        batch_time = (time.time() - start_time) * 1000
        success_rate = (len(results) / len(complaints)) * 100
        
        self.logger.info(
            f"Batch analysis completed in {batch_time:.2f}ms. "
            f"Success rate: {success_rate:.1f}% ({len(results)}/{len(complaints)})"
        )
        
        if failed_complaints:
            self.logger.warning(f"Failed complaints: {failed_complaints}")
        
        return results
    
    def _generate_recommendations(self, 
                                category: str, 
                                priority: str, 
                                escalation: str,
                                metadata: Dict) -> Tuple[str, str]:
        """
        Generate actionable recommendations based on analysis results.
        
        Args:
            category: Predicted category
            priority: Predicted priority
            escalation: Escalation decision
            metadata: Additional processing metadata
            
        Returns:
            Tuple of (recommendation, action_required)
        """
        urgency_detected = metadata.get('urgency_detected', False)
        
        # Base recommendation based on category and priority
        if escalation == "Escalate":
            if priority == "Critical":
                recommendation = "IMMEDIATE ESCALATION: Route to senior management and emergency response team"
                action_required = "Escalate to GM/AGM level, implement emergency protocols"
            else:
                recommendation = "ESCALATION REQUIRED: Forward to department head for review"
                action_required = "Escalate to departmental management within 24 hours"
        
        elif priority == "Critical":
            recommendation = "URGENT ACTION: Immediate response required from relevant department"
            action_required = "Assign to senior staff, respond within 2 hours"
        
        elif priority == "High":
            recommendation = "HIGH PRIORITY: Quick resolution needed"
            action_required = "Assign to experienced staff, resolve within 24 hours"
        
        elif priority == "Medium":
            recommendation = "STANDARD PROCESS: Handle through normal workflow"
            action_required = "Assign to available staff, resolve within 3-5 days"
        
        else:  # Low priority
            recommendation = "LOW PRIORITY: Address when resources available"
            action_required = "Can be handled in routine maintenance cycle"
        
        # Add category-specific guidance
        category_guidance = {
            "Infrastructure": "Coordinate with engineering team for assessment",
            "Safety": "Involve safety officer in resolution process",
            "Staff": "Include HR department in investigation",
            "Cleanliness": "Notify housekeeping and sanitation teams",
            "Food": "Alert catering department and health inspector",
            "Other": "Determine appropriate department for handling"
        }
        
        if category in category_guidance:
            action_required += f". {category_guidance[category]}"
        
        # Add urgency-specific notes
        if urgency_detected:
            recommendation += " (URGENCY DETECTED IN TEXT)"
            
        return recommendation, action_required
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about all models in the pipeline.
        
        Returns:
            Dictionary with model information
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "models": {
                "category_classifier": {
                    "trained": self.category_classifier.is_trained,
                    "classes": self.category_classifier.classes,
                    "model_name": self.category_classifier.model_name
                },
                "priority_classifier": {
                    "trained": self.priority_classifier.is_trained,
                    "classes": self.priority_classifier.classes,
                    "model_name": self.priority_classifier.model_name
                },
                "escalation_classifier": {
                    "trained": self.escalation_classifier.is_trained,
                    "classes": self.escalation_classifier.classes,
                    "model_name": self.escalation_classifier.model_name
                }
            },
            "text_processor": {
                "available": True,
                "railway_keywords": len(self.text_processor.railway_keywords)
            }
        }
    
    def save_models(self, base_path: str) -> None:
        """
        Save all trained models to files.
        
        Args:
            base_path: Base directory to save models
        """
        if not self.is_initialized:
            raise PipelineError("Pipeline not initialized")
        
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save each model
        self.category_classifier.save(os.path.join(base_path, "category_model.pkl"))
        self.priority_classifier.save(os.path.join(base_path, "priority_model.pkl"))
        self.escalation_classifier.save(os.path.join(base_path, "escalation_model.pkl"))
        
        self.logger.info(f"All models saved to {base_path}")
    
    def load_models(self, base_path: str) -> None:
        """
        Load all models from files.
        
        Args:
            base_path: Base directory containing saved models
        """
        import os
        
        # Load each model
        self.category_classifier.load(os.path.join(base_path, "category_model.pkl"))
        self.priority_classifier.load(os.path.join(base_path, "priority_model.pkl"))
        self.escalation_classifier.load(os.path.join(base_path, "escalation_model.pkl"))
        
        self.is_initialized = True
        self.logger.info(f"All models loaded from {base_path}")


def create_pipeline(train_immediately: bool = True) -> ComplaintMLPipeline:
    """
    Factory function to create and optionally initialize the ML pipeline.
    
    Args:
        train_immediately: Whether to train all models immediately
        
    Returns:
        Initialized ComplaintMLPipeline
    """
    pipeline = ComplaintMLPipeline()
    
    if train_immediately:
        pipeline.initialize(train_models=True)
    
    return pipeline