"""
SQLAlchemy Models for Railway Complaint Management System (RCMS)
Comprehensive schema supporting multi-role frontend modules:
- User Interface (Passengers)
- Resolver (Department Staff) 
- Admin Panel (Administrators)
- Analytics (Management/Analysts)
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Enum, Float,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, JSON
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
import enum

from .database import Base

# Enums for standardized values
class UserRole(enum.Enum):
    """User roles in the system."""
    PASSENGER = "passenger"
    RESOLVER = "resolver"
    ADMIN = "admin"
    ANALYST = "analyst"

class ComplaintCategory(enum.Enum):
    """Complaint categories from ML models."""
    INFRASTRUCTURE = "Infrastructure"
    SAFETY = "Safety"
    CLEANLINESS = "Cleanliness"
    STAFF = "Staff"
    FOOD = "Food"
    BOOKING = "Booking"
    OTHER = "Other"

class Priority(enum.Enum):
    """Priority levels for complaints."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class ComplaintStatus(enum.Enum):
    """Status of complaint resolution."""
    SUBMITTED = "submitted"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"

class MediaType(enum.Enum):
    """Types of media attachments."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"

# Base Model with common fields
class BaseModel(Base):
    """Base model with common timestamp fields."""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

# User Management Models
class User(BaseModel):
    """Base user model for all system users."""
    __tablename__ = "users"
    
    # Basic Information
    email = Column(String(255), unique=True, index=True, nullable=False)
    phone = Column(String(20), unique=True, index=True, nullable=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Account Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    phone_verified = Column(Boolean, default=False, nullable=False)
    
    # Role and Permissions
    role = Column(Enum(UserRole), nullable=False, default=UserRole.PASSENGER)
    
    # Metadata
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    
    # Additional Profile Data
    profile_data = Column(JSON, nullable=True)  # Flexible storage for role-specific data
    
    # Relationships
    complaints = relationship("Complaint", back_populates="user", cascade="all, delete-orphan")
    assignments = relationship("ComplaintAssignment", back_populates="resolver")
    
    __table_args__ = (
        Index('idx_user_role_active', 'role', 'is_active'),
        Index('idx_user_email_role', 'email', 'role'),
    )

class Department(BaseModel):
    """Railway departments for complaint resolution."""
    __tablename__ = "departments"
    
    name = Column(String(255), unique=True, nullable=False, index=True)
    code = Column(String(10), unique=True, nullable=False, index=True)  # e.g., "INF", "SAF"
    description = Column(Text, nullable=True)
    
    # Department Head
    head_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    head = relationship("User", foreign_keys=[head_id])
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Category Mapping (which categories this department handles)
    categories = Column(JSON, nullable=True)  # List of complaint categories
    
    # Relationships
    resolvers = relationship("Resolver", back_populates="department")
    assignments = relationship("ComplaintAssignment", back_populates="department")

class Resolver(BaseModel):
    """Department staff members who resolve complaints."""
    __tablename__ = "resolvers"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)
    
    # Staff Information
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    designation = Column(String(255), nullable=False)
    specialization = Column(String(255), nullable=True)  # Area of expertise
    
    # Work Status
    is_available = Column(Boolean, default=True, nullable=False)
    max_assignments = Column(Integer, default=10, nullable=False)
    
    # Performance Metrics
    total_resolved = Column(Integer, default=0, nullable=False)
    avg_resolution_time = Column(Float, nullable=True)  # Hours
    rating = Column(Float, nullable=True)  # Performance rating
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    department = relationship("Department", back_populates="resolvers")
    assignments = relationship("ComplaintAssignment", back_populates="resolver")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'department_id', name='uix_user_department'),
        Index('idx_resolver_available', 'is_available', 'department_id'),
    )

# Complaint Management Models
class Complaint(BaseModel):
    """Main complaint entity."""
    __tablename__ = "complaints"
    
    # User Information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Complaint Content
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text, nullable=False)
    
    # Location Information
    train_number = Column(String(10), nullable=True, index=True)
    station_code = Column(String(10), nullable=True, index=True)
    coach_number = Column(String(10), nullable=True)
    seat_number = Column(String(10), nullable=True)
    
    # Journey Details
    journey_date = Column(DateTime, nullable=True)
    pnr_number = Column(String(20), nullable=True, index=True)
    
    # ML Classification Results
    category = Column(Enum(ComplaintCategory), nullable=False, index=True)
    priority = Column(Enum(Priority), nullable=False, index=True)
    urgency_score = Column(Float, nullable=True)  # ML urgency score 0.0-1.0
    escalation_required = Column(Boolean, default=False, nullable=False)
    
    # ML Confidence Scores
    category_confidence = Column(Float, nullable=True)
    priority_confidence = Column(Float, nullable=True)
    escalation_confidence = Column(Float, nullable=True)
    
    # Status and Resolution
    status = Column(Enum(ComplaintStatus), default=ComplaintStatus.SUBMITTED, nullable=False, index=True)
    
    # Contact Information
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(20), nullable=True)
    preferred_contact_method = Column(String(20), default="email", nullable=False)
    
    # Analytics and Tracking
    source = Column(String(50), default="web", nullable=False)  # web, mobile, api
    resolution_deadline = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    
    # Feedback
    user_rating = Column(Integer, nullable=True)  # 1-5 stars
    user_feedback = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="complaints")
    media_files = relationship("MediaFile", back_populates="complaint", cascade="all, delete-orphan")
    assignments = relationship("ComplaintAssignment", back_populates="complaint", cascade="all, delete-orphan")
    history = relationship("ComplaintHistory", back_populates="complaint", cascade="all, delete-orphan", order_by="ComplaintHistory.created_at.desc()")
    
    __table_args__ = (
        Index('idx_complaint_status_priority', 'status', 'priority'),
        Index('idx_complaint_category_created', 'category', 'created_at'),
        Index('idx_complaint_train_date', 'train_number', 'journey_date'),
    )
    
    @hybrid_property
    def resolution_time_hours(self) -> Optional[float]:
        """Calculate resolution time in hours."""
        if self.resolved_at and self.created_at:
            delta = self.resolved_at - self.created_at
            return delta.total_seconds() / 3600
        return None

class MediaFile(BaseModel):
    """Media attachments for complaints (images, videos, documents)."""
    __tablename__ = "media_files"
    
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False)
    
    # File Information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    media_type = Column(Enum(MediaType), nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # Computer Vision Analysis Results
    ocr_text = Column(Text, nullable=True)  # Extracted text
    vision_analysis = Column(JSON, nullable=True)  # CV analysis results
    
    # Upload Information
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    upload_source = Column(String(50), default="web", nullable=False)
    
    # Relationships
    complaint = relationship("Complaint", back_populates="media_files")
    uploader = relationship("User")
    
    __table_args__ = (
        Index('idx_media_complaint_type', 'complaint_id', 'media_type'),
    )

class ComplaintAssignment(BaseModel):
    """Assignment of complaints to resolvers."""
    __tablename__ = "complaint_assignments"
    
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False)
    resolver_id = Column(Integer, ForeignKey("resolvers.id"), nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)
    assigned_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Assignment Details
    assignment_notes = Column(Text, nullable=True)
    due_date = Column(DateTime, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    complaint = relationship("Complaint", back_populates="assignments")
    resolver = relationship("Resolver", back_populates="assignments")
    department = relationship("Department", back_populates="assignments")
    assigner = relationship("User", foreign_keys=[assigned_by])
    
    __table_args__ = (
        Index('idx_assignment_active', 'is_active', 'resolver_id'),
        Index('idx_assignment_department', 'department_id', 'created_at'),
    )

class ComplaintHistory(BaseModel):
    """History/audit trail of complaint changes."""
    __tablename__ = "complaint_history"
    
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Change Information
    action = Column(String(100), nullable=False, index=True)  # status_change, assignment, comment, etc.
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    comment = Column(Text, nullable=True)
    
    # System Information
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    complaint = relationship("Complaint", back_populates="history")
    user = relationship("User")
    
    __table_args__ = (
        Index('idx_history_complaint_action', 'complaint_id', 'action'),
        Index('idx_history_created', 'created_at'),
    )

# Analytics and Reporting Models
class Analytics(BaseModel):
    """Analytics data for performance monitoring."""
    __tablename__ = "analytics"
    
    # Time Dimensions
    date = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # Categorization
    category = Column(Enum(ComplaintCategory), nullable=True, index=True)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    priority = Column(Enum(Priority), nullable=True, index=True)
    
    # Metrics
    total_complaints = Column(Integer, default=0, nullable=False)
    resolved_complaints = Column(Integer, default=0, nullable=False)
    avg_resolution_time = Column(Float, nullable=True)  # hours
    satisfaction_score = Column(Float, nullable=True)  # 1-5
    escalation_rate = Column(Float, nullable=True)  # percentage
    
    # Additional Metrics
    first_response_time = Column(Float, nullable=True)  # hours
    reopened_count = Column(Integer, default=0, nullable=False)
    overdue_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    department = relationship("Department")
    
    __table_args__ = (
        UniqueConstraint('date', 'period_type', 'category', 'department_id', name='uix_analytics_period'),
        Index('idx_analytics_date_category', 'date', 'category'),
    )

class TrendLog(BaseModel):
    """Logs for trend analysis and pattern detection."""
    __tablename__ = "trend_logs"
    
    # Event Information
    event_type = Column(String(100), nullable=False, index=True)  # complaint_spike, category_trend, etc.
    event_data = Column(JSON, nullable=False)
    
    # Time Information
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    
    # Significance
    confidence_score = Column(Float, nullable=True)  # 0.0-1.0
    impact_level = Column(String(20), nullable=True)  # low, medium, high
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    __table_args__ = (
        Index('idx_trend_type_time', 'event_type', 'start_time'),
    )

class SystemLog(BaseModel):
    """System logs for monitoring and debugging."""
    __tablename__ = "system_logs"
    
    # Log Information
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=False, index=True)
    
    # Context
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=True)
    request_id = Column(String(100), nullable=True, index=True)
    
    # Technical Details
    stack_trace = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User")
    complaint = relationship("Complaint")
    
    __table_args__ = (
        Index('idx_system_log_level_module', 'level', 'module'),
        Index('idx_system_log_created', 'created_at'),
    )