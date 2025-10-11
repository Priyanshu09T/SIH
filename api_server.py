"""
FastAPI Backend for Railway Complaint Management System (RCMS).
Provides REST API endpoints for complaint submission and analysis.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
import logging
from datetime import datetime
import uuid

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.models.complete_ml_models import create_ml_pipeline
from src.preprocessing.text_processor import TextProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Railway Complaint Management System (RCMS)",
    description="AI-powered complaint management with multi-modal processing",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML pipeline instance
ml_pipeline = None
text_processor = TextProcessor()

# Pydantic models for API
class TextAnalysisRequest(BaseModel):
    """Model for text analysis request."""
    text: str = Field(..., description="Text to analyze", min_length=10, max_length=2000)

class ComplaintSubmission(BaseModel):
    """Model for complaint submission."""
    title: str = Field(..., description="Complaint title", min_length=5, max_length=500)
    description: str = Field(..., description="Complaint description", min_length=10, max_length=2000)
    train_number: Optional[str] = Field(None, description="Train number")
    station_code: Optional[str] = Field(None, description="Station code")
    coach_number: Optional[str] = Field(None, description="Coach number")
    seat_number: Optional[str] = Field(None, description="Seat number")
    pnr_number: Optional[str] = Field(None, description="PNR number")
    contact_email: Optional[str] = Field(None, description="Contact email")
    contact_phone: Optional[str] = Field(None, description="Contact phone")
    journey_date: Optional[str] = Field(None, description="Journey date")
    preferred_contact_method: Optional[str] = Field("email", description="Preferred contact method")
    
class ComplaintAnalysisResponse(BaseModel):
    """Model for complaint analysis response."""
    complaint_id: str
    category: Dict[str, Any]
    priority: Dict[str, Any]
    escalation: Dict[str, Any]
    overall_confidence: float
    processed_text: Dict[str, Any]
    timestamp: str
    
class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    ml_pipeline_ready: bool
    timestamp: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize ML pipeline on startup."""
    global ml_pipeline
    try:
        logger.info("Initializing ML pipeline...")
        ml_pipeline = create_ml_pipeline(train_immediately=True)
        logger.info("ML pipeline initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize ML pipeline: {e}")
        raise e

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if ml_pipeline and ml_pipeline.is_trained else "not_ready",
        ml_pipeline_ready=ml_pipeline is not None and ml_pipeline.is_trained,
        timestamp=datetime.now().isoformat()
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Railway Complaint Management System (RCMS)",
        "version": "1.0.0",
        "description": "AI-powered complaint management with multi-modal processing",
        "endpoints": {
            "health": "/health",
            "submit_complaint": "/complaints/submit",
            "analyze_text": "/analyze/text",
            "get_categories": "/categories",
            "docs": "/docs"
        }
    }

# Text analysis endpoint
@app.post("/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    """Analyze complaint text using ML pipeline."""
    if not ml_pipeline or not ml_pipeline.is_trained:
        raise HTTPException(status_code=503, detail="ML pipeline not ready")
    
    try:
        text = request.text
        
        # Validate input
        if len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text too short (minimum 10 characters)")
        
        if len(text) > 2000:
            raise HTTPException(status_code=400, detail="Text too long (maximum 2000 characters)")
        
        # Process text
        processed = text_processor.process(text, extract_features=True)
        
        # Get ML analysis
        analysis = ml_pipeline.analyze_complaint(text)
        
        return {
            "category": analysis["category"]["prediction"],
            "priority": analysis["priority"]["prediction"],
            "confidence": analysis["overall_confidence"],
            "text_features": {
                "cleaned_text": processed.cleaned,
                "word_count": processed.word_count,
                "char_count": processed.char_count,
                "keywords": processed.metadata.get("keywords", {}),
                "urgency_indicators": processed.metadata.get("urgency", {}),
                "has_urgent_indicators": processed.metadata.get("urgency", {}).get("is_urgent", False)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Complaint submission endpoint
@app.post("/complaints/submit", response_model=ComplaintAnalysisResponse)
async def submit_complaint(complaint: ComplaintSubmission):
    """Submit and analyze a new complaint."""
    if not ml_pipeline or not ml_pipeline.is_trained:
        raise HTTPException(status_code=503, detail="ML pipeline not ready")
    
    try:
        # Generate unique complaint ID
        complaint_id = str(uuid.uuid4())
        
        # Combine title and description for analysis
        full_text = f"{complaint.title}. {complaint.description}"
        
        # Process text
        processed = text_processor.process(full_text, extract_features=True)
        
        # Get ML analysis
        analysis = ml_pipeline.analyze_complaint(full_text)
        
        # Create response
        response = ComplaintAnalysisResponse(
            complaint_id=complaint_id,
            category=analysis["category"],
            priority=analysis["priority"],
            escalation=analysis["escalation"],
            overall_confidence=analysis["overall_confidence"],
            processed_text={
                "original": full_text,
                "cleaned": processed.cleaned,
                "word_count": processed.word_count,
                "char_count": processed.char_count,
                "keywords": processed.metadata.get("keywords", {}),
                "urgency_indicators": processed.metadata.get("urgency", {})
            },
            timestamp=datetime.now().isoformat()
        )
        
        # Log complaint submission
        logger.info(f"Complaint submitted: ID={complaint_id}, "
                   f"Category={analysis['category']['prediction']}, "
                   f"Priority={analysis['priority']['prediction']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error submitting complaint: {e}")
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")

# Get available categories
@app.get("/categories")
async def get_categories():
    """Get available complaint categories and priorities."""
    return {
        "categories": [
            "Infrastructure",
            "Cleanliness", 
            "Safety",
            "Staff",
            "Food",
            "Other"
        ],
        "priorities": [
            "Low",
            "Medium", 
            "High",
            "Critical"
        ],
        "escalation_options": [
            "Escalate",
            "No_Escalation"
        ]
    }

# File upload endpoint (for future multi-modal support)
@app.post("/complaints/upload")
async def upload_complaint_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """Upload file-based complaint (images, audio, video)."""
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "audio/wav", "audio/mp3", "video/mp4"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    # For now, return placeholder response
    return {
        "message": "File upload received",
        "filename": file.filename,
        "content_type": file.content_type,
        "description": description,
        "status": "processing_not_implemented",
        "note": "Multi-modal processing will be implemented in the next phase"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/health",
            "/",
            "/complaints/submit",
            "/analyze/text",
            "/categories",
            "/docs"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "contact": "Please contact system administrator"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting RCMS API Server...")
    print("📋 Available endpoints:")
    print("  • GET  /health - Health check")
    print("  • GET  / - API information")
    print("  • POST /complaints/submit - Submit complaint")
    print("  • POST /analyze/text - Analyze text only")
    print("  • GET  /categories - Get available categories")
    print("  • POST /complaints/upload - Upload files")
    print("  • GET  /docs - API documentation")
    print("\n🌐 Server will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )