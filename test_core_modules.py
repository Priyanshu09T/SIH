#!/usr/bin/env python3
"""
Comprehensive test suite for RCMS core modules.
Tests all implemented modules and their integration.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_utilities():
    """Test all utility modules."""
    print("🧪 Testing Utility Modules...")
    
    # Test config
    from src.utils.config import get_settings
    settings = get_settings()
    print(f"✅ Config: {settings.app_name} v{settings.app_version}")
    
    # Test logger
    from src.utils.logger import get_logger, LoggerMixin
    logger = get_logger("test")
    logger.info("Logger test successful")
    print("✅ Logger: Structured logging working")
    
    # Test exceptions
    from src.utils.exceptions import RCMSException, PreprocessingError
    try:
        raise RCMSException("Test exception")
    except RCMSException:
        print("✅ Exceptions: Custom exceptions working")
    
    # Test helpers
    from src.utils.helpers import clean_text, get_file_hash, is_image_file
    cleaned = clean_text("  Test   Text  ")
    assert cleaned == "Test Text"
    print("✅ Helpers: Utility functions working")


def test_text_processor():
    """Test text processing module."""
    print("\n🧪 Testing Text Processor...")
    
    from src.preprocessing.text_processor import TextProcessor, ProcessedText
    
    processor = TextProcessor()
    
    # Test basic processing
    result = processor.process("The train is very dirty and smells terrible!")
    assert isinstance(result, ProcessedText)
    assert result.word_count > 0
    assert len(result.tokens) > 0
    print(f"✅ Basic Processing: {result.word_count} words, {len(result.tokens)} tokens")
    
    # Test railway keyword detection
    railway_text = "The train coach toilet is dirty and the platform is unsafe"
    result = processor.process(railway_text)
    keywords = result.metadata.get("keywords", {})
    assert len(keywords) > 0  # Should detect railway context
    print(f"✅ Keyword Detection: Found {len(keywords)} categories")
    
    # Test urgency detection
    urgent_text = "Emergency! The train caught fire!"
    result = processor.process(urgent_text)
    urgency = result.metadata.get("urgency", {})
    assert urgency.get("level") in ["high", "critical"]
    print(f"✅ Urgency Detection: Level = {urgency.get('level')}")
    
    # Test batch processing
    texts = [
        "The train is dirty",
        "Platform needs cleaning", 
        "Staff was helpful"
    ]
    results = processor.batch_process(texts)
    assert len(results) == len(texts)
    print(f"✅ Batch Processing: Processed {len(results)} texts")


def test_integration():
    """Test integration between modules."""
    print("\n🧪 Testing Module Integration...")
    
    # Test logger + text processor integration
    from src.utils.logger import LoggerMixin
    from src.preprocessing.text_processor import TextProcessor
    
    class IntegratedProcessor(TextProcessor, LoggerMixin):
        def process_with_logging(self, text):
            self.logger.info(f"Processing text: {text[:50]}...")
            result = self.process(text)
            self.logger.info(f"Processed successfully: {result.word_count} words")
            return result
    
    processor = IntegratedProcessor()
    result = processor.process_with_logging("The train toilet is broken and needs urgent repair!")
    assert result.word_count > 0
    print("✅ Integration: Logger + TextProcessor working together")


def run_comprehensive_test():
    """Run all tests."""
    print("🚀 Starting RCMS Core Module Tests")
    print("=" * 50)
    
    try:
        test_utilities()
        test_text_processor()
        test_integration()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! ✅")
        print("✅ Utilities: Config, Logger, Exceptions, Helpers")
        print("✅ Text Processing: Cleaning, Tokenization, Keywords, Urgency")
        print("✅ Integration: Cross-module compatibility")
        print("\n🚉 RCMS Core is ready for development!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)