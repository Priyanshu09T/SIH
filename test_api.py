"""
Test the RCMS API endpoints to validate functionality.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test all API endpoints."""
    print("🧪 Testing RCMS API Endpoints")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()
    
    # Test 2: Root endpoint
    print("2. Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Service: {response.json()['service']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    print()
    
    # Test 3: Get categories
    print("3. Testing Categories Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/categories", timeout=10)
        if response.status_code == 200:
            print("✅ Categories endpoint passed")
            categories = response.json()
            print(f"   Categories: {', '.join(categories['categories'])}")
        else:
            print(f"❌ Categories endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Categories endpoint error: {e}")
    
    print()
    
    # Test 4: Text analysis
    print("4. Testing Text Analysis...")
    test_complaint = "The train toilet is very dirty and smells terrible!"
    try:
        response = requests.post(
            f"{BASE_URL}/analyze/text",
            data={"text": test_complaint},
            timeout=30
        )
        if response.status_code == 200:
            print("✅ Text analysis passed")
            result = response.json()
            analysis = result["analysis"]
            print(f"   Category: {analysis['category']['prediction']} ({analysis['category']['confidence']:.2f})")
            print(f"   Priority: {analysis['priority']['prediction']} ({analysis['priority']['confidence']:.2f})")
            print(f"   Escalate: {'Yes' if analysis['escalation']['should_escalate'] else 'No'}")
        else:
            print(f"❌ Text analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Text analysis error: {e}")
    
    print()
    
    # Test 5: Complaint submission
    print("5. Testing Complaint Submission...")
    complaint_data = {
        "text": "Emergency! There is smoke in coach S3 and passengers are panicking!",
        "passenger_name": "Test User",
        "contact_email": "test@example.com",
        "train_number": "12345",
        "pnr_number": "ABC123"
    }
    try:
        response = requests.post(
            f"{BASE_URL}/complaints/submit",
            json=complaint_data,
            timeout=30
        )
        if response.status_code == 200:
            print("✅ Complaint submission passed")
            result = response.json()
            print(f"   Complaint ID: {result['complaint_id']}")
            print(f"   Category: {result['category']['prediction']}")
            print(f"   Priority: {result['priority']['prediction']}")
            print(f"   Should Escalate: {'Yes' if result['escalation']['should_escalate'] else 'No'}")
        else:
            print(f"❌ Complaint submission failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Complaint submission error: {e}")
    
    print()
    print("🎉 API Testing Complete!")

if __name__ == "__main__":
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    test_api_endpoints()