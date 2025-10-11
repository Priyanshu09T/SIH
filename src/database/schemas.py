"""
Pydantic schemas for API request/response validation and serialization
Provides data validation, serialization, and API documentation support
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, ConfigDict, Field, validator
from enum import Enum

# Import enums from models
from .models import UserRole, ComplaintCategory, Priority, ComplaintStatus, MediaType

# Base Schemas
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: datetime

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(from_attributes=True)

# User Schemas
class UserBase(BaseSchema):
    """Base user schema."""
    email: EmailStr
    phone: Optional[str] = None
    full_name: str
    role: UserRole = UserRole.PASSENGER

class UserCreate(UserBase):
    """Schema for user creation."""
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserUpdate(BaseSchema):
    """Schema for user updates."""
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    profile_data: Optional[Dict[str, Any]] = None

class UserLogin(BaseSchema):
    """Schema for user login."""
    email: EmailStr
    password: str

class UserResponse(UserBase, TimestampMixin):
    """Schema for user response."""
    id: int
    is_active: bool
    is_verified: bool
    email_verified: bool
    phone_verified: bool
    last_login: Optional[datetime] = None
    login_count: int = 0
    profile_data: Optional[Dict[str, Any]] = None

class UserProfile(BaseSchema):
    """Schema for user profile information."""
    id: int
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    last_login: Optional[datetime] = None

# Department Schemas
class DepartmentBase(BaseSchema):
    """Base department schema."""
    name: str
    code: str = Field(..., max_length=10, description="Department code (e.g., INF, SAF)")
    description: Optional[str] = None
    categories: Optional[List[ComplaintCategory]] = None

class DepartmentCreate(DepartmentBase):
    """Schema for department creation."""
    head_id: Optional[int] = None

class DepartmentUpdate(BaseSchema):
    """Schema for department updates."""
    name: Optional[str] = None
    description: Optional[str] = None
    head_id: Optional[int] = None
    is_active: Optional[bool] = None
    categories: Optional[List[ComplaintCategory]] = None

class DepartmentResponse(DepartmentBase, TimestampMixin):
    """Schema for department response."""
    id: int
    head_id: Optional[int] = None
    is_active: bool

# Resolver Schemas
class ResolverBase(BaseSchema):
    """Base resolver schema."""
    employee_id: str
    designation: str
    specialization: Optional[str] = None
    max_assignments: int = 10

class ResolverCreate(ResolverBase):
    """Schema for resolver creation."""
    user_id: int
    department_id: int

class ResolverUpdate(BaseSchema):
    """Schema for resolver updates."""
    designation: Optional[str] = None
    specialization: Optional[str] = None
    is_available: Optional[bool] = None
    max_assignments: Optional[int] = None

class ResolverResponse(ResolverBase, TimestampMixin):
    """Schema for resolver response."""
    id: int
    user_id: int
    department_id: int
    is_available: bool
    total_resolved: int = 0
    avg_resolution_time: Optional[float] = None
    rating: Optional[float] = None

# Complaint Schemas
class ComplaintBase(BaseSchema):
    """Base complaint schema."""
    title: str = Field(..., max_length=500)
    description: str
    train_number: Optional[str] = Field(None, max_length=10)
    station_code: Optional[str] = Field(None, max_length=10)
    coach_number: Optional[str] = Field(None, max_length=10)
    seat_number: Optional[str] = Field(None, max_length=10)
    journey_date: Optional[datetime] = None
    pnr_number: Optional[str] = Field(None, max_length=20)
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None
    preferred_contact_method: str = "email"

class ComplaintCreate(ComplaintBase):
    """Schema for complaint creation."""
    source: str = "web"

class ComplaintUpdate(BaseSchema):
    """Schema for complaint updates."""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ComplaintStatus] = None
    priority: Optional[Priority] = None
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None

class ComplaintMLResult(BaseSchema):
    """Schema for ML classification results."""
    category: ComplaintCategory
    priority: Priority
    urgency_score: Optional[float] = None
    escalation_required: bool = False
    category_confidence: Optional[float] = None
    priority_confidence: Optional[float] = None
    escalation_confidence: Optional[float] = None

class ComplaintResponse(ComplaintBase, TimestampMixin):
    """Schema for complaint response."""
    id: int
    user_id: int
    category: ComplaintCategory
    priority: Priority
    urgency_score: Optional[float] = None
    escalation_required: bool = False
    status: ComplaintStatus
    resolution_deadline: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    user_rating: Optional[int] = None
    user_feedback: Optional[str] = None

class ComplaintDetail(ComplaintResponse):
    """Detailed complaint schema with relationships."""
    user: UserProfile
    assignments: List["AssignmentResponse"] = []
    media_files: List["MediaFileResponse"] = []
    history: List["HistoryResponse"] = []

# Media File Schemas
class MediaFileBase(BaseSchema):
    """Base media file schema."""
    filename: str
    original_filename: str
    media_type: MediaType
    mime_type: str

class MediaFileCreate(MediaFileBase):
    """Schema for media file creation."""
    complaint_id: int
    file_path: str
    file_size: int
    upload_source: str = "web"

class MediaFileResponse(MediaFileBase, TimestampMixin):
    """Schema for media file response."""
    id: int
    complaint_id: int
    file_size: int
    ocr_text: Optional[str] = None
    vision_analysis: Optional[Dict[str, Any]] = None
    uploaded_by: int
    upload_source: str

# Assignment Schemas
class AssignmentBase(BaseSchema):
    """Base assignment schema."""
    assignment_notes: Optional[str] = None
    due_date: Optional[datetime] = None

class AssignmentCreate(AssignmentBase):
    """Schema for assignment creation."""
    complaint_id: int
    resolver_id: int
    department_id: int

class AssignmentUpdate(BaseSchema):
    """Schema for assignment updates."""
    assignment_notes: Optional[str] = None
    due_date: Optional[datetime] = None
    is_active: Optional[bool] = None

class AssignmentResponse(AssignmentBase, TimestampMixin):
    """Schema for assignment response."""
    id: int
    complaint_id: int
    resolver_id: int
    department_id: int
    assigned_by: int
    is_active: bool
    accepted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# History Schemas
class HistoryCreate(BaseSchema):
    """Schema for history creation."""
    complaint_id: int
    action: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    comment: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class HistoryResponse(TimestampMixin):
    """Schema for history response."""
    id: int
    complaint_id: int
    user_id: Optional[int] = None
    action: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    comment: Optional[str] = None

# Analytics Schemas
class AnalyticsBase(BaseSchema):
    """Base analytics schema."""
    date: datetime
    period_type: str = Field(..., regex="^(daily|weekly|monthly)$")
    category: Optional[ComplaintCategory] = None
    department_id: Optional[int] = None
    priority: Optional[Priority] = None

class AnalyticsCreate(AnalyticsBase):
    """Schema for analytics creation."""
    total_complaints: int = 0
    resolved_complaints: int = 0
    avg_resolution_time: Optional[float] = None
    satisfaction_score: Optional[float] = None
    escalation_rate: Optional[float] = None
    first_response_time: Optional[float] = None
    reopened_count: int = 0
    overdue_count: int = 0

class AnalyticsResponse(AnalyticsCreate, TimestampMixin):
    """Schema for analytics response."""
    id: int

# Dashboard Schemas
class DashboardStats(BaseSchema):
    """Schema for dashboard statistics."""
    total_complaints: int
    pending_complaints: int
    resolved_complaints: int
    critical_complaints: int
    avg_resolution_time: Optional[float] = None
    satisfaction_score: Optional[float] = None
    top_categories: List[Dict[str, Any]] = []
    recent_trends: List[Dict[str, Any]] = []

class CategoryStats(BaseSchema):
    """Schema for category statistics."""
    category: ComplaintCategory
    count: int
    percentage: float
    avg_resolution_time: Optional[float] = None
    satisfaction_score: Optional[float] = None

# API Response Schemas
class APIResponse(BaseSchema):
    """Standard API response schema."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

class PaginatedResponse(BaseSchema):
    """Schema for paginated responses."""
    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool

# Update forward references
ComplaintDetail.model_rebuild()
AssignmentResponse.model_rebuild()
MediaFileResponse.model_rebuild()
HistoryResponse.model_rebuild()