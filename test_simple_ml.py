"""
Test the simplified ML models.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_category_classifier():
    """Test the category classifier."""
    print("🧪 Testing Category Classifier...")
    
    try:
        from src.models.category_classifier_simple import CategoryClassifier, create_category_classifier
        
        # Test initialization
        classifier = CategoryClassifier()
        assert not classifier.is_trained
        print("✅ Classifier initialization successful")
        
        # Test training
        classifier.train()
        assert classifier.is_trained
        print("✅ Classifier training successful")
        
        # Test prediction
        test_complaints = [
            "The train toilet is very dirty and smells bad",
            "Platform roof is leaking during rain", 
            "Staff conductor was very rude to passengers",
            "Food quality in pantry car is terrible",
            "Emergency brake system not working properly",
            "WiFi not working in coach"
        ]
        
        expected_categories = [
            "Cleanliness",
            "Infrastructure", 
            "Staff",
            "Food",
            "Safety",
            "Other"
        ]
        
        for i, complaint in enumerate(test_complaints):
            result = classifier.predict(complaint)
            
            print(f"✅ Text: '{complaint[:50]}...'")
            print(f"   Predicted: {result['prediction']} (confidence: {result['confidence']:.2f})")
            print(f"   Expected: {expected_categories[i]}")
            
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["prediction"] in classifier.CATEGORIES
            assert 0 <= result["confidence"] <= 1
        
        print("✅ All predictions successful")
        
        # Test batch prediction
        batch_results = classifier.predict_batch(test_complaints)
        assert len(batch_results) == len(test_complaints)
        print("✅ Batch prediction successful")
        
        # Test factory function
        quick_classifier = create_category_classifier(train_immediately=True)
        assert quick_classifier.is_trained
        print("✅ Factory function successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Category classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Simplified ML Models Test")
    print("=" * 60)
    
    success = test_category_classifier()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All ML model tests passed!")
        print("✅ Category Classification working")
        print("✅ Training and prediction successful")
        print("✅ Ready for integration with API")
    else:
        print("❌ Some tests failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)