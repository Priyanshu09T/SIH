"""
Simple database test to verify our CRUD operations work
Tests database functionality without requiring the API server
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Simple test without full database integration
def test_simple_database():
    """Test basic database operations."""
    print("🔍 Testing Simple Database Operations...")
    
    try:
        # Test our simple database
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/rcms_test.db")
        if not db_path.exists():
            print("❌ Database not found. Please run scripts/init_db.py first")
            return False
        
        # Connect and test
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Test data exists
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"✅ Users in database: {user_count}")
        
        cursor.execute("SELECT COUNT(*) FROM departments")
        dept_count = cursor.fetchone()[0]
        print(f"✅ Departments in database: {dept_count}")
        
        cursor.execute("SELECT COUNT(*) FROM complaints")
        complaint_count = cursor.fetchone()[0]
        print(f"✅ Complaints in database: {complaint_count}")
        
        # Test inserting a complaint
        cursor.execute("""
            INSERT INTO complaints (complaint_number, user_id, title, description, train_number)
            VALUES (?, ?, ?, ?, ?)
        """, ("TEST001", 1, "Test complaint", "This is a test complaint for AC not working", "12345"))
        
        conn.commit()
        print("✅ Test complaint inserted successfully")
        
        # Verify insertion
        cursor.execute("SELECT * FROM complaints WHERE complaint_number = 'TEST001'")
        complaint = cursor.fetchone()
        if complaint:
            print(f"✅ Test complaint retrieved: {complaint[2]}")  # title
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def test_ml_models():
    """Test ML models without API."""
    print("\n🔍 Testing ML Models...")
    
    try:
        from src.models.complete_ml_models import CompleteMLPipeline
        
        # Initialize pipeline
        pipeline = CompleteMLPipeline()
        print("✅ ML Pipeline initialized")
        
        # Train models
        print("📚 Training models...")
        pipeline.train()
        print("✅ Models trained successfully")
        
        # Test predictions
        test_texts = [
            "AC not working in coach B2",
            "Food quality is very poor",
            "Station is very dirty",
            "App is crashing frequently"
        ]
        
        for text in test_texts:
            result = pipeline.predict(text)
            print(f"✅ Prediction for '{text[:30]}...': {result['category']} ({result['priority']})")
        
        return True
        
    except Exception as e:
        print(f"❌ ML models test error: {e}")
        return False

def test_text_processing():
    """Test text processing module."""
    print("\n🔍 Testing Text Processing...")
    
    try:
        from src.preprocessing.text_processor import TextProcessor
        
        processor = TextProcessor()
        print("✅ Text processor initialized")
        
        test_texts = [
            "The AC in coach B2 is not working properly!",
            "खाना बहुत खराब है",  # Hindi text
            "Station platform needs cleaning urgently"
        ]
        
        for text in test_texts:
            processed = processor.process_text(text)
            print(f"✅ Processed '{text[:30]}...' -> {len(processed['cleaned_text'])} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test error: {e}")
        return False

def main():
    """Run all standalone tests."""
    print("🚀 Starting RCMS Standalone Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Database Operations", test_simple_database()))
    test_results.append(("ML Models", test_ml_models()))
    test_results.append(("Text Processing", test_text_processing()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"📈 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Core functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)