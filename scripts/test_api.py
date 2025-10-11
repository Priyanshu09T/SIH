"""
Test FastAPI backend with database integration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import json
from datetime import datetime

def test_api_endpoints():
    """Test FastAPI endpoints with database integration."""
    base_url = "http://localhost:8000"
    
    print("🌐 Testing FastAPI Backend with Database...")
    print("=" * 60)
    
    # Test data
    complaint_data = {
        "title": "Train delay and poor service",
        "description": "Train 12345 was delayed by 2 hours and the staff was not helpful. Food quality was also poor.",
        "train_number": "12345",
        "station_code": "NDLS",
        "source": "web"
    }
    
    try:
        # Test 1: Health check
        print("\n🏥 Test 1: API Health Check")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ API is healthy")
            print(f"   📊 Status: {health_data['status']}")
            print(f"   🕐 Timestamp: {health_data['timestamp']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test 2: Submit complaint
        print("\n📝 Test 2: Submit Complaint")
        response = requests.post(f"{base_url}/complaints/", json=complaint_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Complaint submitted successfully")
            print(f"   📋 Complaint ID: {result.get('complaint_id')}")
            print(f"   🎯 Category: {result.get('category')}")
            print(f"   🔥 Priority: {result.get('priority')}")
            print(f"   📊 Urgency Score: {result.get('urgency_score', 'N/A')}")
            print(f"   ⚠️ Escalation Required: {result.get('escalation_required', False)}")
            
            complaint_id = result.get('complaint_id')
        else:
            print(f"   ❌ Complaint submission failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
        
        # Test 3: Get complaint status
        print("\n🔍 Test 3: Get Complaint Status")
        response = requests.get(f"{base_url}/complaints/{complaint_id}")
        if response.status_code == 200:
            status_data = response.json()
            print(f"   ✅ Retrieved complaint status")
            print(f"   📋 ID: {status_data.get('id')}")
            print(f"   📝 Title: {status_data.get('title')}")
            print(f"   📊 Status: {status_data.get('status')}")
            print(f"   🎯 Category: {status_data.get('category')}")
            print(f"   🔥 Priority: {status_data.get('priority')}")
        else:
            print(f"   ❌ Status retrieval failed: {response.status_code}")
            return False
        
        # Test 4: List all complaints
        print("\n📋 Test 4: List All Complaints")
        response = requests.get(f"{base_url}/complaints/")
        if response.status_code == 200:
            complaints = response.json()
            print(f"   ✅ Retrieved {len(complaints)} complaints")
            for i, complaint in enumerate(complaints[:3], 1):  # Show first 3
                print(f"   {i}. ID: {complaint.get('id')} - {complaint.get('title')[:50]}...")
                print(f"      Status: {complaint.get('status')} | Priority: {complaint.get('priority')}")
        else:
            print(f"   ❌ Complaints listing failed: {response.status_code}")
            return False
        
        # Test 5: ML Prediction endpoint
        print("\n🤖 Test 5: ML Prediction")
        ml_data = {
            "text": "The toilet in coach S3 is very dirty and smells bad. Please clean it immediately."
        }
        response = requests.post(f"{base_url}/predict/", json=ml_data)
        if response.status_code == 200:
            prediction = response.json()
            print(f"   ✅ ML prediction successful")
            print(f"   🎯 Category: {prediction.get('category')}")
            print(f"   🔥 Priority: {prediction.get('priority')}")
            print(f"   📊 Urgency Score: {prediction.get('urgency_score', 'N/A')}")
            print(f"   ⚠️ Escalation: {prediction.get('escalation_required', False)}")
            
            # Show confidence scores if available
            if 'category_confidence' in prediction:
                print(f"   📈 Category Confidence: {prediction['category_confidence']:.2f}")
            if 'priority_confidence' in prediction:
                print(f"   📈 Priority Confidence: {prediction['priority_confidence']:.2f}")
        else:
            print(f"   ❌ ML prediction failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
        
        # Test 6: Analytics endpoint
        print("\n📊 Test 6: Analytics")
        response = requests.get(f"{base_url}/analytics/")
        if response.status_code == 200:
            analytics = response.json()
            print(f"   ✅ Analytics retrieved successfully")
            print(f"   📋 Total Complaints: {analytics.get('total_complaints', 0)}")
            print(f"   ⏳ Pending: {analytics.get('pending_complaints', 0)}")
            print(f"   ✅ Resolved: {analytics.get('resolved_complaints', 0)}")
            print(f"   🔥 Critical: {analytics.get('critical_complaints', 0)}")
            
            # Category distribution
            categories = analytics.get('category_distribution', [])
            if categories:
                print(f"   🎯 Top Categories:")
                for i, cat in enumerate(categories[:3], 1):
                    print(f"      {i}. {cat.get('category')}: {cat.get('count')} complaints")
        else:
            print(f"   ❌ Analytics retrieval failed: {response.status_code}")
        
        print("=" * 60)
        print("✅ All API tests completed successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: FastAPI server is not running!")
        print("💡 Please start the server first:")
        print("   cd P:\\SIH && P:/SIH/venv/Scripts/python.exe -m uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error during API testing: {e}")
        return False

def start_api_server():
    """Start the FastAPI server."""
    print("🚀 Starting FastAPI Server...")
    print("💡 Run this command in another terminal:")
    print("   cd P:\\SIH && P:/SIH/venv/Scripts/python.exe -m uvicorn src.api.main:app --reload")
    print("💡 Then run this test script again")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 API docs will be at: http://localhost:8000/docs")

if __name__ == "__main__":
    print("🧪 FastAPI + Database Integration Test")
    print(f"🕐 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we should start server or run tests
    if len(sys.argv) > 1 and sys.argv[1] == "--start-server":
        start_api_server()
    else:
        success = test_api_endpoints()
        if not success:
            print("\n💡 If the server is not running, use:")
            print("   python scripts/test_api.py --start-server")
            sys.exit(1)