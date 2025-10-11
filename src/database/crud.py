"""
CRUD operations for database entities
Provides create, read, update, delete operations with proper error handling
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import and_, or_, desc, asc, func, text
from passlib.context import CryptContext

from .models import (
    User, Department, Resolver, Complaint, MediaFile,
    ComplaintAssignment, ComplaintHistory, Analytics, TrendLog,
    SystemLog, UserRole, ComplaintCategory, Priority, ComplaintStatus
)
from .schemas import (
    UserCreate, UserUpdate, DepartmentCreate, DepartmentUpdate,
    ResolverCreate, ResolverUpdate, ComplaintCreate, ComplaintUpdate,
    MediaFileCreate, AssignmentCreate, AssignmentUpdate,
    HistoryCreate, AnalyticsCreate
)

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class CRUDException(Exception):
    """Custom exception for CRUD operations."""
    pass

class UserCRUD:
    """CRUD operations for User entity."""
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create(db: Session, user_data: UserCreate) -> User:
        """Create a new user."""
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                or_(User.email == user_data.email, User.phone == user_data.phone)
            ).first()
            
            if existing_user:
                raise CRUDException("User with this email or phone already exists")
            
            # Hash password
            hashed_password = UserCRUD.get_password_hash(user_data.password)
            
            # Create user
            db_user = User(
                email=user_data.email,
                password_hash=hashed_password,
                phone=user_data.phone,
                full_name=user_data.full_name,
                role=user_data.role,
                is_active=True,
                is_verified=False
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"Created user: {db_user.email}")
            return db_user
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error creating user: {str(e)}")
            raise CRUDException("User creation failed: duplicate data")
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error creating user: {str(e)}")
            raise CRUDException("User creation failed")
    
    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_active_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Get active users with pagination."""
        return db.query(User).filter(User.is_active == True).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_role(db: Session, role: UserRole, skip: int = 0, limit: int = 100) -> List[User]:
        """Get users by role."""
        return db.query(User).filter(User.role == role).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Update user."""
        try:
            db_user = UserCRUD.get_by_id(db, user_id)
            if not db_user:
                return None
            
            # Update fields
            for field, value in user_data.model_dump(exclude_unset=True).items():
                setattr(db_user, field, value)
            
            db_user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"Updated user: {db_user.email}")
            return db_user
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating user: {str(e)}")
            raise CRUDException("User update failed")
    
    @staticmethod
    def delete(db: Session, user_id: int) -> bool:
        """Delete user (soft delete by setting is_active=False)."""
        try:
            db_user = UserCRUD.get_by_id(db, user_id)
            if not db_user:
                return False
            
            db_user.is_active = False
            db_user.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Deactivated user: {db_user.email}")
            return True
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error deactivating user: {str(e)}")
            return False
    
    @staticmethod
    def authenticate(db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = UserCRUD.get_by_email(db, email)
        if not user or not UserCRUD.verify_password(password, user.password_hash):
            return None
        
        # Update login info
        user.last_login = datetime.utcnow()
        user.login_count += 1
        db.commit()
        
        return user

class DepartmentCRUD:
    """CRUD operations for Department entity."""
    
    @staticmethod
    def create(db: Session, dept_data: DepartmentCreate) -> Department:
        """Create a new department."""
        try:
            # Check if department code already exists
            existing_dept = db.query(Department).filter(Department.code == dept_data.code).first()
            if existing_dept:
                raise CRUDException("Department with this code already exists")
            
            db_dept = Department(**dept_data.model_dump())
            db.add(db_dept)
            db.commit()
            db.refresh(db_dept)
            
            logger.info(f"Created department: {db_dept.name}")
            return db_dept
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error creating department: {str(e)}")
            raise CRUDException("Department creation failed: duplicate data")
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error creating department: {str(e)}")
            raise CRUDException("Department creation failed")
    
    @staticmethod
    def get_by_id(db: Session, dept_id: int) -> Optional[Department]:
        """Get department by ID."""
        return db.query(Department).options(joinedload(Department.head)).filter(Department.id == dept_id).first()
    
    @staticmethod
    def get_by_code(db: Session, code: str) -> Optional[Department]:
        """Get department by code."""
        return db.query(Department).filter(Department.code == code).first()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Department]:
        """Get all departments."""
        return db.query(Department).filter(Department.is_active == True).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_category(db: Session, category: ComplaintCategory) -> List[Department]:
        """Get departments that handle specific category."""
        return db.query(Department).filter(
            Department.categories.contains([category]),
            Department.is_active == True
        ).all()
    
    @staticmethod
    def update(db: Session, dept_id: int, dept_data: DepartmentUpdate) -> Optional[Department]:
        """Update department."""
        try:
            db_dept = DepartmentCRUD.get_by_id(db, dept_id)
            if not db_dept:
                return None
            
            for field, value in dept_data.model_dump(exclude_unset=True).items():
                setattr(db_dept, field, value)
            
            db_dept.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_dept)
            
            logger.info(f"Updated department: {db_dept.name}")
            return db_dept
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating department: {str(e)}")
            raise CRUDException("Department update failed")

class ComplaintCRUD:
    """CRUD operations for Complaint entity."""
    
    @staticmethod
    def create(db: Session, complaint_data: ComplaintCreate, user_id: int) -> Complaint:
        """Create a new complaint."""
        try:
            # Generate unique complaint number
            complaint_number = ComplaintCRUD._generate_complaint_number(db)
            
            db_complaint = Complaint(
                **complaint_data.model_dump(),
                user_id=user_id,
                complaint_number=complaint_number,
                status=ComplaintStatus.SUBMITTED,
                priority=Priority.MEDIUM  # Default, will be updated by ML
            )
            
            db.add(db_complaint)
            db.commit()
            db.refresh(db_complaint)
            
            # Log complaint creation
            HistoryCRUD.create(db, HistoryCreate(
                complaint_id=db_complaint.id,
                action="COMPLAINT_CREATED",
                comment="Complaint submitted successfully"
            ), user_id)
            
            logger.info(f"Created complaint: {db_complaint.complaint_number}")
            return db_complaint
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating complaint: {str(e)}")
            raise CRUDException("Complaint creation failed")
    
    @staticmethod
    def _generate_complaint_number(db: Session) -> str:
        """Generate unique complaint number."""
        today = datetime.now()
        prefix = f"RCMS{today.year}{today.month:02d}{today.day:02d}"
        
        # Get count of complaints today
        count = db.query(Complaint).filter(
            func.date(Complaint.created_at) == today.date()
        ).count()
        
        return f"{prefix}{count + 1:04d}"
    
    @staticmethod
    def get_by_id(db: Session, complaint_id: int) -> Optional[Complaint]:
        """Get complaint by ID with all relationships."""
        return db.query(Complaint).options(
            joinedload(Complaint.user),
            selectinload(Complaint.assignments).joinedload(ComplaintAssignment.resolver),
            selectinload(Complaint.media_files),
            selectinload(Complaint.history)
        ).filter(Complaint.id == complaint_id).first()
    
    @staticmethod
    def get_by_number(db: Session, complaint_number: str) -> Optional[Complaint]:
        """Get complaint by complaint number."""
        return db.query(Complaint).filter(Complaint.complaint_number == complaint_number).first()
    
    @staticmethod
    def get_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Complaint]:
        """Get complaints by user."""
        return db.query(Complaint).filter(
            Complaint.user_id == user_id
        ).order_by(desc(Complaint.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_status(db: Session, status: ComplaintStatus, skip: int = 0, limit: int = 100) -> List[Complaint]:
        """Get complaints by status."""
        return db.query(Complaint).filter(
            Complaint.status == status
        ).order_by(desc(Complaint.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_priority(db: Session, priority: Priority, skip: int = 0, limit: int = 100) -> List[Complaint]:
        """Get complaints by priority."""
        return db.query(Complaint).filter(
            Complaint.priority == priority
        ).order_by(desc(Complaint.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_pending_complaints(db: Session, skip: int = 0, limit: int = 100) -> List[Complaint]:
        """Get pending complaints."""
        return db.query(Complaint).filter(
            Complaint.status.in_([
                ComplaintStatus.SUBMITTED,
                ComplaintStatus.ASSIGNED,
                ComplaintStatus.IN_PROGRESS
            ])
        ).order_by(desc(Complaint.priority), desc(Complaint.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_overdue_complaints(db: Session) -> List[Complaint]:
        """Get overdue complaints."""
        now = datetime.utcnow()
        return db.query(Complaint).filter(
            and_(
                Complaint.resolution_deadline < now,
                Complaint.status.in_([
                    ComplaintStatus.SUBMITTED,
                    ComplaintStatus.ASSIGNED,
                    ComplaintStatus.IN_PROGRESS
                ])
            )
        ).all()
    
    @staticmethod
    def update(db: Session, complaint_id: int, complaint_data: ComplaintUpdate, user_id: int) -> Optional[Complaint]:
        """Update complaint."""
        try:
            db_complaint = ComplaintCRUD.get_by_id(db, complaint_id)
            if not db_complaint:
                return None
            
            # Track changes for history
            changes = []
            for field, new_value in complaint_data.model_dump(exclude_unset=True).items():
                old_value = getattr(db_complaint, field)
                if old_value != new_value:
                    changes.append({
                        "field": field,
                        "old_value": str(old_value),
                        "new_value": str(new_value)
                    })
                    setattr(db_complaint, field, new_value)
            
            if changes:
                db_complaint.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(db_complaint)
                
                # Log changes
                for change in changes:
                    HistoryCRUD.create(db, HistoryCreate(
                        complaint_id=complaint_id,
                        action=f"FIELD_UPDATED_{change['field'].upper()}",
                        old_value=change['old_value'],
                        new_value=change['new_value']
                    ), user_id)
                
                logger.info(f"Updated complaint: {db_complaint.complaint_number}")
            
            return db_complaint
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating complaint: {str(e)}")
            raise CRUDException("Complaint update failed")
    
    @staticmethod
    def get_dashboard_stats(db: Session) -> Dict[str, Any]:
        """Get dashboard statistics."""
        try:
            total_complaints = db.query(Complaint).count()
            pending_complaints = db.query(Complaint).filter(
                Complaint.status.in_([
                    ComplaintStatus.SUBMITTED,
                    ComplaintStatus.ASSIGNED,
                    ComplaintStatus.IN_PROGRESS
                ])
            ).count()
            
            resolved_complaints = db.query(Complaint).filter(
                Complaint.status == ComplaintStatus.RESOLVED
            ).count()
            
            critical_complaints = db.query(Complaint).filter(
                and_(
                    Complaint.priority == Priority.CRITICAL,
                    Complaint.status != ComplaintStatus.RESOLVED
                )
            ).count()
            
            # Average resolution time (in hours)
            avg_resolution = db.query(
                func.avg(func.extract('epoch', Complaint.resolved_at - Complaint.created_at) / 3600)
            ).filter(Complaint.resolved_at.isnot(None)).scalar() or 0
            
            # Category statistics
            category_stats = db.query(
                Complaint.category,
                func.count(Complaint.id).label('count')
            ).group_by(Complaint.category).all()
            
            top_categories = [
                {"category": stat[0].value, "count": stat[1]}
                for stat in category_stats[:5]
            ]
            
            return {
                "total_complaints": total_complaints,
                "pending_complaints": pending_complaints,
                "resolved_complaints": resolved_complaints,
                "critical_complaints": critical_complaints,
                "avg_resolution_time": round(avg_resolution, 2),
                "top_categories": top_categories
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting dashboard stats: {str(e)}")
            return {}

class HistoryCRUD:
    """CRUD operations for ComplaintHistory entity."""
    
    @staticmethod
    def create(db: Session, history_data: HistoryCreate, user_id: Optional[int] = None) -> ComplaintHistory:
        """Create a new history entry."""
        try:
            db_history = ComplaintHistory(
                **history_data.model_dump(),
                user_id=user_id
            )
            
            db.add(db_history)
            db.commit()
            db.refresh(db_history)
            
            return db_history
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating history entry: {str(e)}")
            raise CRUDException("History creation failed")
    
    @staticmethod
    def get_by_complaint(db: Session, complaint_id: int) -> List[ComplaintHistory]:
        """Get history entries for a complaint."""
        return db.query(ComplaintHistory).filter(
            ComplaintHistory.complaint_id == complaint_id
        ).order_by(desc(ComplaintHistory.created_at)).all()

# Export all CRUD classes
__all__ = [
    "UserCRUD", "DepartmentCRUD", "ComplaintCRUD", "HistoryCRUD",
    "CRUDException"
]