"""
Test script to verify database CRUD operations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import sqlite3
from datetime import datetime

def test_database_operations():
    """Test basic database operations."""
    db_path = "data/rcms_test.db"
    
    print("🧪 Testing Database CRUD Operations...")
    print("=" * 50)
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test 1: Read existing data
        print("\n📖 Test 1: Reading existing data")
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        print(f"   Found {len(users)} users:")
        for user in users:
            print(f"   - ID: {user[0]}, Email: {user[1]}, Name: {user[3]}, Role: {user[4]}")
        
        cursor.execute("SELECT * FROM departments")
        departments = cursor.fetchall()
        print(f"   Found {len(departments)} departments:")
        for dept in departments:
            print(f"   - ID: {dept[0]}, Code: {dept[2]}, Name: {dept[1]}")
        
        # Test 2: Create a new complaint
        print("\n✏️ Test 2: Creating a new complaint")
        complaint_data = {
            'complaint_number': f"RCMS2025101101001",
            'user_id': 2,  # passenger user
            'title': 'AC not working in coach B2',
            'description': 'The air conditioning system in coach B2 is not functioning properly. Temperature is very high.',
            'category': 'technical_issue',
            'priority': 'high',
            'status': 'submitted',
            'train_number': '12345',
            'station_code': 'NDLS'
        }
        
        cursor.execute("""
            INSERT INTO complaints 
            (complaint_number, user_id, title, description, category, priority, status, train_number, station_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            complaint_data['complaint_number'],
            complaint_data['user_id'],
            complaint_data['title'],
            complaint_data['description'],
            complaint_data['category'],
            complaint_data['priority'],
            complaint_data['status'],
            complaint_data['train_number'],
            complaint_data['station_code']
        ))
        
        conn.commit()
        complaint_id = cursor.lastrowid
        print(f"   ✅ Created complaint with ID: {complaint_id}")
        print(f"   📋 Complaint Number: {complaint_data['complaint_number']}")
        print(f"   📝 Title: {complaint_data['title']}")
        
        # Test 3: Read the created complaint
        print("\n🔍 Test 3: Reading created complaint")
        cursor.execute("""
            SELECT c.*, u.full_name, u.email 
            FROM complaints c 
            JOIN users u ON c.user_id = u.id 
            WHERE c.id = ?
        """, (complaint_id,))
        
        complaint = cursor.fetchone()
        if complaint:
            print(f"   📋 Complaint Details:")
            print(f"   - ID: {complaint[0]}")
            print(f"   - Number: {complaint[1]}")
            print(f"   - Title: {complaint[3]}")
            print(f"   - Status: {complaint[7]}")
            print(f"   - Priority: {complaint[6]}")
            print(f"   - User: {complaint[11]} ({complaint[12]})")
            print(f"   - Created: {complaint[10]}")
        
        # Test 4: Update complaint status
        print("\n🔄 Test 4: Updating complaint status")
        cursor.execute("""
            UPDATE complaints 
            SET status = 'in_progress', priority = 'critical'
            WHERE id = ?
        """, (complaint_id,))
        
        conn.commit()
        print(f"   ✅ Updated complaint {complaint_id} status to 'in_progress'")
        print(f"   ✅ Updated complaint {complaint_id} priority to 'critical'")
        
        # Test 5: Verify update
        print("\n✅ Test 5: Verifying update")
        cursor.execute("SELECT status, priority FROM complaints WHERE id = ?", (complaint_id,))
        result = cursor.fetchone()
        print(f"   📊 Current Status: {result[0]}")
        print(f"   📊 Current Priority: {result[1]}")
        
        # Test 6: Database statistics
        print("\n📊 Test 6: Database Statistics")
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM departments")
        dept_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM complaints")
        complaint_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT status, COUNT(*) FROM complaints GROUP BY status")
        status_stats = cursor.fetchall()
        
        cursor.execute("SELECT priority, COUNT(*) FROM complaints GROUP BY priority")
        priority_stats = cursor.fetchall()
        
        print(f"   👥 Total Users: {user_count}")
        print(f"   🏢 Total Departments: {dept_count}")
        print(f"   📋 Total Complaints: {complaint_count}")
        print(f"   📈 Status Distribution:")
        for status, count in status_stats:
            print(f"      - {status}: {count}")
        print(f"   🔥 Priority Distribution:")
        for priority, count in priority_stats:
            print(f"      - {priority}: {count}")
        
        # Test 7: Complex query - Join with user data
        print("\n🔗 Test 7: Complex Query - Complaints with User Info")
        cursor.execute("""
            SELECT 
                c.complaint_number,
                c.title,
                c.status,
                c.priority,
                u.full_name,
                u.email,
                c.created_at
            FROM complaints c
            JOIN users u ON c.user_id = u.id
            ORDER BY c.created_at DESC
            LIMIT 5
        """)
        
        complaints = cursor.fetchall()
        print(f"   📋 Recent Complaints:")
        for complaint in complaints:
            print(f"      - {complaint[0]}: {complaint[1]}")
            print(f"        👤 User: {complaint[4]} ({complaint[5]})")
            print(f"        📊 Status: {complaint[2]} | Priority: {complaint[3]}")
            print(f"        🕐 Created: {complaint[6]}")
            print()
        
        print("=" * 50)
        print("✅ All database tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during database testing: {e}")
        return False
    finally:
        conn.close()
    
    return True

def test_database_schema():
    """Test database schema and structure."""
    db_path = "data/rcms_test.db"
    
    print("\n🏗️ Testing Database Schema...")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall() if table[0] != 'sqlite_sequence']
        
        print(f"📋 Tables in database: {tables}")
        
        for table in tables:
            print(f"\n🔍 Table: {table}")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            print(f"   📊 Columns ({len(columns)}):")
            for col in columns:
                print(f"      - {col[1]} ({col[2]}) {'NOT NULL' if col[3] else ''} {'PRIMARY KEY' if col[5] else ''}")
        
        print("=" * 50)
        print("✅ Schema verification completed!")
        
    except Exception as e:
        print(f"❌ Error during schema testing: {e}")
        return False
    finally:
        conn.close()
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Database Testing Suite...")
    print(f"🕐 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test schema first
    schema_success = test_database_schema()
    
    if schema_success:
        # Test CRUD operations
        crud_success = test_database_operations()
        
        if crud_success:
            print("\n🎉 All tests passed! Database is working correctly.")
        else:
            print("\n❌ CRUD tests failed!")
            sys.exit(1)
    else:
        print("\n❌ Schema tests failed!")
        sys.exit(1)