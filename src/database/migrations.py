"""
Database migrations and initialization utilities
Handles database schema creation, updates, and data seeding
"""

import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path

from .database import Database
from .models import (
    Base, User, Department, Resolver, Complaint, MediaFile,
    ComplaintAssignment, ComplaintHistory, Analytics, TrendLog,
    SystemLog, UserRole, ComplaintCategory, Priority, ComplaintStatus
)

logger = logging.getLogger(__name__)

class DatabaseMigration:
    """Handle database migrations and initialization."""
    
    def __init__(self, database: Database):
        self.database = database
        self.engine = database.engine
    
    def create_all_tables(self) -> bool:
        """Create all database tables."""
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self.engine)
            logger.info("All tables created successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {str(e)}")
            return False
    
    def drop_all_tables(self) -> bool:
        """Drop all database tables (use with caution)."""
        try:
            logger.warning("Dropping all database tables...")
            Base.metadata.drop_all(bind=self.engine)
            logger.info("All tables dropped successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error dropping tables: {str(e)}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()
    
    def get_existing_tables(self) -> List[str]:
        """Get list of existing tables."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def initialize_database(self, drop_existing: bool = False) -> bool:
        """Initialize database with tables and seed data."""
        try:
            if drop_existing:
                self.drop_all_tables()
            
            # Create tables
            if not self.create_all_tables():
                return False
            
            # Seed initial data
            if not self.seed_initial_data():
                return False
                
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False
    
    def seed_initial_data(self) -> bool:
        """Seed initial data into the database."""
        try:
            with self.database.get_session() as session:
                # Check if data already exists
                if session.query(Department).count() > 0:
                    logger.info("Database already contains data, skipping seed")
                    return True
                
                # Create departments
                departments = [
                    Department(
                        name="Information Technology",
                        code="IT",
                        description="Technical issues, app problems, website issues",
                        categories=[
                            ComplaintCategory.TECHNICAL_ISSUE,
                            ComplaintCategory.WEBSITE_APP
                        ]
                    ),
                    Department(
                        name="Safety & Security",
                        code="SAF",
                        description="Safety concerns, security issues, emergency situations",
                        categories=[
                            ComplaintCategory.SAFETY_SECURITY,
                            ComplaintCategory.STAFF_BEHAVIOR
                        ]
                    ),
                    Department(
                        name="Infrastructure",
                        code="INF",
                        description="Station facilities, train amenities, cleanliness",
                        categories=[
                            ComplaintCategory.CLEANLINESS,
                            ComplaintCategory.FACILITIES
                        ]
                    ),
                    Department(
                        name="Customer Service",
                        code="CS",
                        description="Ticketing, refunds, general inquiries",
                        categories=[
                            ComplaintCategory.TICKETING,
                            ComplaintCategory.REFUND,
                            ComplaintCategory.OTHER
                        ]
                    ),
                    Department(
                        name="Catering Services",
                        code="CAT",
                        description="Food quality, catering complaints",
                        categories=[
                            ComplaintCategory.FOOD_CATERING
                        ]
                    )
                ]
                
                for dept in departments:
                    session.add(dept)
                
                # Create admin user
                admin_user = User(
                    email="admin@railway.gov.in",
                    password_hash="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/vvVo3j9wS",  # password: admin123
                    full_name="System Administrator",
                    role=UserRole.ADMIN,
                    is_active=True,
                    is_verified=True,
                    email_verified=True
                )
                session.add(admin_user)
                
                # Create sample resolver user
                resolver_user = User(
                    email="resolver@railway.gov.in",
                    password_hash="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/vvVo3j9wS",  # password: admin123
                    full_name="John Resolver",
                    role=UserRole.RESOLVER,
                    is_active=True,
                    is_verified=True,
                    email_verified=True
                )
                session.add(resolver_user)
                
                # Create sample passenger user
                passenger_user = User(
                    email="passenger@example.com",
                    password_hash="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/vvVo3j9wS",  # password: admin123
                    full_name="Jane Passenger",
                    role=UserRole.PASSENGER,
                    is_active=True,
                    is_verified=True,
                    email_verified=True
                )
                session.add(passenger_user)
                
                session.commit()
                
                # Create resolver profile for IT department
                it_dept = session.query(Department).filter_by(code="IT").first()
                if it_dept:
                    resolver = Resolver(
                        user_id=resolver_user.id,
                        department_id=it_dept.id,
                        employee_id="EMP001",
                        designation="Senior IT Specialist",
                        specialization="Mobile Apps, Web Applications",
                        max_assignments=15,
                        is_available=True
                    )
                    session.add(resolver)
                
                # Set IT department head
                it_dept.head_id = resolver_user.id
                
                session.commit()
                
                logger.info("Initial data seeded successfully")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error seeding initial data: {str(e)}")
            return False
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Create a backup of the database."""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backup_rcms_{timestamp}.sql"
            
            # This is a simplified backup for SQLite
            # For production, use proper backup tools
            if "sqlite" in str(self.engine.url):
                import shutil
                db_path = str(self.engine.url).replace("sqlite:///", "")
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backed up to {backup_path}")
                return True
            else:
                logger.warning("Backup not implemented for this database type")
                return False
                
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return False
    
    def get_migration_status(self) -> dict:
        """Get current migration status."""
        try:
            existing_tables = self.get_existing_tables()
            expected_tables = [
                "users", "departments", "resolvers", "complaints",
                "media_files", "complaint_assignments", "complaint_history",
                "analytics", "trend_logs", "system_logs"
            ]
            
            missing_tables = [table for table in expected_tables if table not in existing_tables]
            extra_tables = [table for table in existing_tables if table not in expected_tables]
            
            with self.database.get_session() as session:
                user_count = session.query(User).count()
                dept_count = session.query(Department).count()
                complaint_count = session.query(Complaint).count()
            
            return {
                "tables_created": len(existing_tables),
                "expected_tables": len(expected_tables),
                "missing_tables": missing_tables,
                "extra_tables": extra_tables,
                "user_count": user_count,
                "department_count": dept_count,
                "complaint_count": complaint_count,
                "status": "complete" if not missing_tables else "incomplete"
            }
            
        except Exception as e:
            logger.error(f"Error getting migration status: {str(e)}")
            return {"status": "error", "error": str(e)}

def init_database(database_url: Optional[str] = None, drop_existing: bool = False) -> bool:
    """Initialize database with all tables and seed data."""
    try:
        db = Database(database_url)
        migration = DatabaseMigration(db)
        return migration.initialize_database(drop_existing)
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    # Script can be run directly for database initialization
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration utility")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--backup", help="Create database backup")
    parser.add_argument("--database-url", help="Database URL")
    
    args = parser.parse_args()
    
    if args.database_url:
        db = Database(args.database_url)
    else:
        db = Database()
    
    migration = DatabaseMigration(db)
    
    if args.status:
        status = migration.get_migration_status()
        print(f"Migration Status: {status}")
    
    elif args.backup:
        if migration.backup_database(args.backup):
            print(f"Backup created successfully")
        else:
            print("Backup failed")
    
    elif args.init:
        if migration.initialize_database(args.drop):
            print("Database initialized successfully")
        else:
            print("Database initialization failed")
    
    else:
        print("Use --help for available options")