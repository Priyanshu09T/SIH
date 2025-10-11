"""
Simple database initialization script without dependencies
"""

import sqlite3
import os
from pathlib import Path

def create_simple_database():
    """Create a simple SQLite database for testing."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Database file path
    db_path = data_dir / "rcms_test.db"
    
    # Remove existing database
    if db_path.exists():
        os.remove(db_path)
    
    # Create connection
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Create users table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT DEFAULT 'passenger',
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create departments table
        cursor.execute("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT UNIQUE NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create complaints table
        cursor.execute("""
            CREATE TABLE complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_number TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT DEFAULT 'other',
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'submitted',
                train_number TEXT,
                station_code TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Insert sample data
        cursor.execute("""
            INSERT INTO departments (name, code, description) VALUES
            ('Information Technology', 'IT', 'Technical issues and app problems'),
            ('Safety & Security', 'SAF', 'Safety concerns and security issues'),
            ('Customer Service', 'CS', 'Ticketing and general inquiries')
        """)
        
        cursor.execute("""
            INSERT INTO users (email, password_hash, full_name, role) VALUES
            ('admin@railway.gov.in', '$2b$12$hash', 'System Administrator', 'admin'),
            ('passenger@example.com', '$2b$12$hash', 'Jane Passenger', 'passenger')
        """)
        
        conn.commit()
        print(f"✅ Database created successfully at: {db_path}")
        print("📊 Sample data inserted")
        
        # Show database info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📋 Tables created: {[table[0] for table in tables]}")
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"👥 Users: {user_count}")
        
        cursor.execute("SELECT COUNT(*) FROM departments") 
        dept_count = cursor.fetchone()[0]
        print(f"🏢 Departments: {dept_count}")
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False
    finally:
        conn.close()
    
    return True

if __name__ == "__main__":
    print("🚀 Initializing RCMS Database...")
    if create_simple_database():
        print("✅ Database initialization completed successfully!")
    else:
        print("❌ Database initialization failed!")