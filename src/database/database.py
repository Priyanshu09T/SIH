"""
Database configuration and connection management for RCMS
Supports SQLite for development and PostgreSQL for production
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging

from src.utils.settings import get_settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base Model
Base = declarative_base()

class Database:
    """Database manager for RCMS with support for multiple database backends."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self._setup_database()
    
    def _setup_database(self):
        """Initialize database connection and session factory."""
        
        # Determine database URL based on environment
        if self.settings.database_url:
            # Use provided database URL (production)
            database_url = self.settings.database_url
            logger.info("Using configured database URL")
            
        elif self.settings.environment == "production":
            # PostgreSQL for production
            database_url = (
                f"postgresql://{self.settings.db_user}:{self.settings.db_password}"
                f"@{self.settings.db_host}:{self.settings.db_port}/{self.settings.db_name}"
            )
            logger.info("Using PostgreSQL database for production")
            
        else:
            # SQLite for development/testing
            db_path = os.path.join(self.settings.data_dir, "rcms.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f"sqlite:///{db_path}"
            logger.info(f"Using SQLite database: {db_path}")
        
        # Create engine with appropriate settings
        if database_url.startswith("sqlite"):
            # SQLite specific settings
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=self.settings.debug
            )
        else:
            # PostgreSQL settings
            self.engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=self.settings.debug
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database connection established successfully")
    
    def create_tables(self):
        """Create all database tables."""
        from .models import Base  # Import Base from models
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        """Drop all database tables (for testing/reset)."""
        from .models import Base  # Import Base from models
        
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def reset_database(self):
        """Reset database by dropping and recreating all tables."""
        self.drop_tables()
        self.create_tables()
        logger.info("Database reset completed")

# Global database instance
_database = None

def get_database() -> Database:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database session.
    Used with Depends() in API endpoints.
    """
    database = get_database()
    db = database.get_session()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with tables for application startup."""
    database = get_database()
    database.create_tables()
    return database