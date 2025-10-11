"""
Very simple ML test without complex imports.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_basic_ml():
    """Test basic ML functionality without complex imports."""
    print("🧪 Testing Basic ML Pipeline...")
    
    try:
        # Test data
        train_texts = [
            "train toilet dirty smells bad",
            "platform roof leaking rain",
            "staff conductor rude passengers", 
            "food quality pantry terrible",
            "emergency brake not working",
            "wifi not working coach"
        ]
        
        train_labels = [
            "Cleanliness",
            "Infrastructure",
            "Staff", 
            "Food",
            "Safety",
            "Other"
        ]
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Train
        pipeline.fit(train_texts, train_labels)
        print("✅ Pipeline trained successfully")
        
        # Test predictions
        test_texts = [
            "The toilet is very dirty",
            "Platform needs repair", 
            "Conductor was rude",
            "Food was terrible",
            "Emergency brake failed",
            "Internet not working"
        ]
        
        for text in test_texts:
            prediction = pipeline.predict([text])[0]
            probabilities = pipeline.predict_proba([text])[0]
            confidence = max(probabilities)
            
            print(f"✅ '{text}' → {prediction} (conf: {confidence:.2f})")
        
        print("✅ All predictions successful")
        return True
        
    except Exception as e:
        print(f"❌ Basic ML test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_processing():
    """Test text processing independently."""
    print("\n🧪 Testing Text Processing...")
    
    try:
        from src.preprocessing.text_processor import TextProcessor
        
        processor = TextProcessor()
        
        test_text = "The train toilet is very dirty and smells terrible!"
        result = processor.process(test_text)
        
        print(f"✅ Original: '{result.original}'")
        print(f"✅ Cleaned: '{result.cleaned}'")
        print(f"✅ Tokens: {result.tokens}")
        print(f"✅ Word count: {result.word_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests."""
    print("🚀 Simple ML Tests")
    print("=" * 40)
    
    success = True
    
    if not test_basic_ml():
        success = False
    
    if not test_text_processing():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All basic ML tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)