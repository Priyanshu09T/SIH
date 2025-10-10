# Simple test script to validate text processing
import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ProcessedText:
    original: str
    cleaned: str
    tokens: List[str]
    word_count: int
    char_count: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SimpleTextProcessor:
    def __init__(self):
        self.stop_words = {'the', 'is', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
        self.railway_keywords = {
            'infrastructure': ['train', 'coach', 'seat', 'toilet', 'platform'],
            'cleanliness': ['dirty', 'clean', 'smell', 'garbage'],
            'safety': ['danger', 'accident', 'emergency'],
            'staff': ['conductor', 'staff', 'service'],
            'food': ['food', 'meal', 'water']
        }
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip('.,!?-').strip()
    
    def tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]
    
    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        found = {}
        text_lower = text.lower()
        for category, keywords in self.railway_keywords.items():
            matches = [k for k in keywords if k in text_lower]
            if matches:
                found[category] = matches
        return found
    
    def detect_urgency(self, text: str) -> str:
        urgent_words = ['emergency', 'urgent', 'critical', 'serious', 'problem']
        text_lower = text.lower()
        for word in urgent_words:
            if word in text_lower:
                return 'high'
        return 'low'
    
    def process(self, text: str) -> ProcessedText:
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        result = ProcessedText(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            word_count=len(tokens),
            char_count=len(cleaned),
            metadata={
                'keywords': self.extract_keywords(cleaned),
                'urgency': self.detect_urgency(cleaned),
                'has_railway_context': bool(self.extract_keywords(cleaned))
            }
        )
        return result

# Test it
if __name__ == "__main__":
    processor = SimpleTextProcessor()
    test_text = "The train toilet is very dirty and smells bad! This is urgent!"
    result = processor.process(test_text)
    
    print("✅ Simple Text Processor Test")
    print(f"Original: {result.original}")
    print(f"Cleaned: {result.cleaned}")
    print(f"Tokens: {result.tokens}")
    print(f"Word count: {result.word_count}")
    print(f"Keywords: {result.metadata['keywords']}")
    print(f"Urgency: {result.metadata['urgency']}")
    print(f"Railway context: {result.metadata['has_railway_context']}")