"""
Simple test script to validate ML models functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that we can import all modules."""
    print("🧪 Testing ML Model Imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        print("✅ Scientific libraries imported successfully")
        
        # Test our base model
        from src.models.base_model import BaseModel, SklearnModel, ModelPrediction
        print("✅ Base model classes imported")
        
        # Test text processor
        from src.preprocessing.text_processor import TextProcessor
        processor = TextProcessor()
        print("✅ Text processor imported and initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_pipeline():
    """Test a simple sklearn pipeline."""
    print("\n🧪 Testing Simple Pipeline...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        
        # Create simple data
        texts = [
            "Train is dirty and smells bad",
            "Platform needs cleaning",
            "Staff was very helpful",
            "Food quality is poor",
            "Emergency brake not working",
            "AC not functioning properly"
        ]
        
        labels = [
            "Cleanliness",
            "Cleanliness", 
            "Staff",
            "Food",
            "Safety",
            "Infrastructure"
        ]
        
        # Create simple pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=100, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Train
        pipeline.fit(texts, labels)
        print("✅ Pipeline trained successfully")
        
        # Predict
        test_text = "The toilet is very dirty"
        prediction = pipeline.predict([test_text])[0]
        probabilities = pipeline.predict_proba([test_text])[0]
        
        print(f"✅ Prediction: {prediction}")
        print(f"✅ Max probability: {max(probabilities):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_processor_integration():
    """Test text processor with ML pipeline."""
    print("\n🧪 Testing Text Processor + ML Integration...")
    
    try:
        from src.preprocessing.text_processor import TextProcessor
        
        processor = TextProcessor()
        
        # Test text processing
        text = "The train toilet is extremely dirty and smells terrible!"
        processed = processor.process(text)
        
        print(f"✅ Text processed: '{processed.cleaned}'")
        print(f"✅ Tokens: {processed.tokens}")
        print(f"✅ Keywords: {list(processed.metadata.get('keywords', {}).keys())}")
        print(f"✅ Urgency: {processed.metadata.get('urgency', {}).get('level')}")
        
        # Test with sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Use processed text for ML
        vectorizer = TfidfVectorizer(max_features=100)
        
        # Create sample data using processed text
        sample_texts = [
            processor.process("Train is dirty").cleaned,
            processor.process("Staff was helpful").cleaned,
            processor.process("Food is bad").cleaned
        ]
        
        # Fit vectorizer
        vectorizer.fit(sample_texts)
        vectors = vectorizer.transform([processed.cleaned])
        
        print(f"✅ Text vectorized: shape {vectors.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processor integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting ML Models Validation")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test simple pipeline
    if not test_simple_pipeline():
        success = False
    
    # Test text processor integration
    if not test_text_processor_integration():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All ML validation tests passed!")
        print("Ready to implement ML models properly.")
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)