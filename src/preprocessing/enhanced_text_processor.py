"""
Enhanced Railway-Specific Feature Engineering
Improved text processing with better railway terminology and urgency detection
"""

import re
import string
from typing import List, Dict, Set, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTextFeatures:
    """Enhanced text processing results with detailed features."""
    original: str
    cleaned: str
    tokens: List[str]
    word_count: int
    char_count: int
    
    # Railway-specific features
    railway_keywords: List[str]
    infrastructure_indicators: List[str]
    safety_indicators: List[str]
    cleanliness_indicators: List[str]
    staff_indicators: List[str]
    food_indicators: List[str]
    
    # Urgency and escalation features
    urgency_score: float
    urgency_indicators: List[str]
    escalation_keywords: List[str]
    
    # Sentiment and emotion indicators
    negative_sentiment_words: List[str]
    emotion_indicators: List[str]
    
    # Contextual features
    has_emergency_context: bool
    has_complaint_context: bool
    has_suggestion_context: bool
    
    # Derived flags
    is_urgent: bool
    needs_immediate_attention: bool
    suggests_escalation: bool


class EnhancedTextProcessor:
    """Enhanced text processor with comprehensive railway-specific features."""
    
    def __init__(self):
        self.railway_keywords = self._load_railway_keywords()
        self.category_indicators = self._load_category_indicators()
        self.urgency_indicators = self._load_urgency_indicators()
        self.escalation_keywords = self._load_escalation_keywords()
        self.sentiment_words = self._load_sentiment_words()
        self.emotion_indicators = self._load_emotion_indicators()
        self.context_patterns = self._load_context_patterns()
    
    def _load_railway_keywords(self) -> Set[str]:
        """Load comprehensive railway-specific keywords."""
        return {
            # Infrastructure
            'train', 'coach', 'compartment', 'bogie', 'wagon', 'engine', 'locomotive',
            'platform', 'station', 'track', 'rail', 'signal', 'crossing', 'bridge',
            'tunnel', 'overhead', 'wire', 'pantograph', 'catenary',
            
            # Facilities
            'toilet', 'washroom', 'bathroom', 'lavatory', 'restroom', 'berth', 'seat',
            'window', 'door', 'ac', 'fan', 'light', 'charging', 'socket', 'wifi',
            'pantry', 'canteen', 'catering', 'dining',
            
            # Staff
            'conductor', 'tte', 'guard', 'driver', 'pilot', 'station master', 'announcer',
            'vendor', 'cleaner', 'staff', 'employee', 'personnel', 'officer',
            
            # Services
            'ticket', 'reservation', 'booking', 'pnr', 'waitlist', 'rac', 'refund',
            'tatkal', 'premium', 'express', 'passenger', 'local', 'goods',
            
            # Technical
            'brake', 'coupling', 'journal', 'axle', 'wheel', 'suspension', 'buffer',
            'vacuum', 'air', 'pressure', 'speed', 'acceleration', 'deceleration'
        }
    
    def _load_category_indicators(self) -> Dict[str, Set[str]]:
        """Load category-specific indicator words."""
        return {
            'Infrastructure': {
                # Physical structure issues
                'broken', 'damaged', 'cracked', 'leaking', 'loose', 'stuck', 'jammed',
                'not working', 'malfunctioning', 'defective', 'faulty', 'repair needed',
                'maintenance', 'replacement', 'fixing', 'renovation',
                
                # Specific infrastructure components
                'platform', 'track', 'signal', 'bridge', 'tunnel', 'overhead wire',
                'catenary', 'pantograph', 'coupling', 'brake system', 'door mechanism',
                'window', 'seat', 'berth', 'coach', 'bogie', 'axle', 'wheel',
                
                # Infrastructure problems
                'derailment', 'track fault', 'signal failure', 'power failure',
                'mechanical failure', 'structural damage', 'wear and tear'
            },
            
            'Safety': {
                # Immediate danger
                'emergency', 'danger', 'hazard', 'risk', 'unsafe', 'threatening',
                'accident', 'injury', 'hurt', 'bleeding', 'unconscious', 'fall',
                'collision', 'crash', 'fire', 'smoke', 'explosion', 'gas leak',
                
                # Security threats
                'theft', 'robbery', 'pickpocket', 'harassment', 'assault', 'fight',
                'suspicious', 'unattended', 'bomb', 'weapon', 'knife', 'violence',
                
                # Safety equipment issues
                'emergency brake', 'fire extinguisher', 'first aid', 'safety chain',
                'alarm', 'warning', 'evacuation', 'escape route', 'exit blocked',
                
                # Medical emergencies
                'medical emergency', 'heart attack', 'seizure', 'breathing problem',
                'allergic reaction', 'food poisoning', 'illness', 'fever'
            },
            
            'Cleanliness': {
                # Hygiene issues
                'dirty', 'filthy', 'unclean', 'unhygienic', 'contaminated', 'polluted',
                'stained', 'soiled', 'grimy', 'dusty', 'muddy', 'greasy',
                
                # Specific cleanliness problems
                'garbage', 'trash', 'litter', 'waste', 'rubbish', 'debris',
                'vomit', 'urine', 'feces', 'spit', 'blood stains', 'food spills',
                
                # Pests and infestations
                'cockroach', 'rat', 'mouse', 'insect', 'bug', 'pest', 'infestation',
                'spider', 'ant', 'fly', 'mosquito',
                
                # Smell and odor
                'smell', 'odor', 'stink', 'foul', 'bad smell', 'rotten', 'sewage',
                'toilet smell', 'garbage smell',
                
                # Cleaning supplies missing
                'no soap', 'no tissue', 'no water', 'no cleaning', 'dustbin full'
            },
            
            'Staff': {
                # Behavior issues
                'rude', 'impolite', 'unprofessional', 'disrespectful', 'arrogant',
                'unhelpful', 'negligent', 'careless', 'irresponsible', 'lazy',
                
                # Misconduct
                'bribe', 'corruption', 'extortion', 'fraud', 'cheating', 'lying',
                'harassment', 'discrimination', 'favoritism', 'abuse of power',
                
                # Service failures
                'absent', 'missing', 'unavailable', 'not responding', 'ignoring',
                'refusing', 'denying', 'delaying', 'poor service',
                
                # Positive feedback
                'helpful', 'polite', 'courteous', 'professional', 'excellent service',
                'thank you', 'appreciate', 'grateful', 'good job'
            },
            
            'Food': {
                # Food quality issues
                'stale', 'spoiled', 'rotten', 'expired', 'bad taste', 'tasteless',
                'undercooked', 'overcooked', 'cold food', 'burnt', 'raw',
                
                # Food safety issues
                'food poisoning', 'stomach ache', 'nausea', 'vomiting', 'diarrhea',
                'contaminated', 'insects in food', 'hair in food', 'foreign object',
                
                # Service issues
                'overpriced', 'expensive', 'no change', 'wrong order', 'missing item',
                'delayed service', 'rude vendor', 'unhygienic preparation',
                
                # Food items
                'meal', 'dinner', 'lunch', 'breakfast', 'tea', 'coffee', 'water',
                'snacks', 'biscuit', 'bread', 'rice', 'dal', 'curry', 'vegetable'
            }
        }
    
    def _load_urgency_indicators(self) -> Dict[str, float]:
        """Load urgency indicators with weights."""
        return {
            # Critical urgency (weight 1.0)
            'emergency': 1.0, 'urgent': 1.0, 'immediate': 1.0, 'asap': 1.0,
            'help': 1.0, 'sos': 1.0, 'danger': 1.0, 'critical': 1.0,
            
            # High urgency (weight 0.8)
            'serious': 0.8, 'severe': 0.8, 'major': 0.8, 'important': 0.8,
            'priority': 0.8, 'quickly': 0.8, 'fast': 0.8, 'soon': 0.8,
            
            # Medium urgency (weight 0.6)
            'please': 0.6, 'need': 0.6, 'required': 0.6, 'necessary': 0.6,
            'issue': 0.6, 'problem': 0.6, 'concern': 0.6, 'matter': 0.6,
            
            # Time-sensitive words (weight 0.7)
            'now': 0.7, 'today': 0.7, 'immediately': 0.9, 'right now': 0.9,
            'this instant': 0.9, 'without delay': 0.8
        }
    
    def _load_escalation_keywords(self) -> Set[str]:
        """Load keywords that suggest escalation need."""
        return {
            # Authority escalation
            'manager', 'supervisor', 'senior officer', 'higher authority',
            'complaint', 'formal complaint', 'escalate', 'report',
            
            # Legal/media threats
            'legal action', 'court', 'police', 'media', 'newspaper', 'social media',
            'consumer court', 'railway board', 'minister',
            
            # Multiple incidents
            'repeatedly', 'always', 'every time', 'multiple times', 'again and again',
            'continuous', 'ongoing', 'persistent',
            
            # Severity indicators
            'unacceptable', 'outrageous', 'disgraceful', 'shameful', 'terrible',
            'horrible', 'worst', 'disgusting', 'pathetic'
        }
    
    def _load_sentiment_words(self) -> Dict[str, List[str]]:
        """Load sentiment indicator words."""
        return {
            'negative': [
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic',
                'worst', 'hate', 'angry', 'frustrated', 'annoyed', 'disappointed',
                'upset', 'furious', 'mad', 'irritated', 'outraged', 'shocked'
            ],
            'positive': [
                'good', 'excellent', 'great', 'wonderful', 'amazing', 'fantastic',
                'perfect', 'love', 'happy', 'satisfied', 'pleased', 'grateful',
                'thankful', 'appreciate', 'impressed', 'outstanding'
            ]
        }
    
    def _load_emotion_indicators(self) -> Set[str]:
        """Load emotion-indicating words and phrases."""
        return {
            'fear', 'scared', 'terrified', 'worried', 'anxious', 'nervous',
            'panic', 'panicking', 'shocked', 'stunned', 'surprised',
            'confused', 'bewildered', 'lost', 'helpless', 'desperate',
            'frustrated', 'irritated', 'annoyed', 'angry', 'furious',
            'disappointed', 'sad', 'upset', 'hurt', 'betrayed'
        }
    
    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """Load context-indicating patterns."""
        return {
            'emergency': [
                r'emergency.*', r'urgent.*help', r'immediate.*attention',
                r'danger.*', r'fire.*', r'accident.*', r'injury.*'
            ],
            'complaint': [
                r'complain.*', r'complaint.*', r'issue.*', r'problem.*',
                r'not.*satisfied', r'disappointed.*', r'unacceptable.*'
            ],
            'suggestion': [
                r'suggest.*', r'suggestion.*', r'recommend.*', r'should.*',
                r'could.*improve', r'better.*if', r'would.*like'
            ]
        }
    
    def _calculate_urgency_score(self, text: str, urgency_words: List[str]) -> float:
        """Calculate urgency score based on found urgency indicators."""
        if not urgency_words:
            return 0.0
        
        total_score = 0.0
        for word in urgency_words:
            weight = self.urgency_indicators.get(word.lower(), 0.5)
            total_score += weight
        
        # Normalize by text length and cap at 1.0
        normalized_score = min(total_score / max(len(text.split()), 1), 1.0)
        return normalized_score
    
    def _find_category_indicators(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Find category-specific indicators in tokens."""
        found_indicators = {category: [] for category in self.category_indicators}
        
        # Check individual tokens
        for token in tokens:
            for category, indicators in self.category_indicators.items():
                if token.lower() in indicators:
                    found_indicators[category].append(token)
        
        # Check multi-word phrases in original text
        text_lower = ' '.join(tokens).lower()
        for category, indicators in self.category_indicators.items():
            for indicator in indicators:
                if len(indicator.split()) > 1:  # Multi-word indicator
                    if indicator in text_lower:
                        found_indicators[category].append(indicator)
        
        return found_indicators
    
    def _detect_context(self, text: str) -> Dict[str, bool]:
        """Detect different contexts in the text."""
        contexts = {}
        
        for context_type, patterns in self.context_patterns.items():
            contexts[f"has_{context_type}_context"] = any(
                re.search(pattern, text.lower()) for pattern in patterns
            )
        
        return contexts
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with railway-specific preprocessing."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve important railway abbreviations and codes
        railway_abbrevs = {
            'tte': 'ticket_collector',
            'ac': 'air_conditioning', 
            'pnr': 'passenger_name_record',
            'rac': 'reservation_against_cancellation',
            'sms': 'text_message',
            'gps': 'location_system',
            'cctv': 'security_camera',
            'rpm': 'railway_protection_force'
        }
        
        for abbrev, expansion in railway_abbrevs.items():
            text = re.sub(r'\b' + abbrev + r'\b', expansion, text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive punctuation but preserve sentence structure
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization preserving railway-specific terms."""
        if not text:
            return []
        
        # Split on whitespace and punctuation, but preserve hyphenated terms
        tokens = re.findall(r'\b[\w-]+\b', text.lower())
        
        # Filter out very short tokens and pure numbers
        tokens = [token for token in tokens 
                 if len(token) > 1 and not token.isdigit()]
        
        return tokens
    
    def process(self, text: str, extract_features: bool = True) -> EnhancedTextFeatures:
        """Process text with enhanced feature extraction."""
        if not text:
            return EnhancedTextFeatures(
                original="", cleaned="", tokens=[], word_count=0, char_count=0,
                railway_keywords=[], infrastructure_indicators=[], safety_indicators=[],
                cleanliness_indicators=[], staff_indicators=[], food_indicators=[],
                urgency_score=0.0, urgency_indicators=[], escalation_keywords=[],
                negative_sentiment_words=[], emotion_indicators=[],
                has_emergency_context=False, has_complaint_context=False, has_suggestion_context=False,
                is_urgent=False, needs_immediate_attention=False, suggests_escalation=False
            )
        
        # Basic processing
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        word_count = len(tokens)
        char_count = len(cleaned)
        
        if not extract_features:
            return EnhancedTextFeatures(
                original=text, cleaned=cleaned, tokens=tokens,
                word_count=word_count, char_count=char_count,
                railway_keywords=[], infrastructure_indicators=[], safety_indicators=[],
                cleanliness_indicators=[], staff_indicators=[], food_indicators=[],
                urgency_score=0.0, urgency_indicators=[], escalation_keywords=[],
                negative_sentiment_words=[], emotion_indicators=[],
                has_emergency_context=False, has_complaint_context=False, has_suggestion_context=False,
                is_urgent=False, needs_immediate_attention=False, suggests_escalation=False
            )
        
        # Extract railway-specific keywords
        railway_keywords = [token for token in tokens if token in self.railway_keywords]
        
        # Find category indicators
        category_indicators = self._find_category_indicators(tokens)
        
        # Find urgency indicators
        urgency_words = [token for token in tokens if token in self.urgency_indicators]
        urgency_score = self._calculate_urgency_score(cleaned, urgency_words)
        
        # Find escalation keywords
        escalation_words = [token for token in tokens if token in self.escalation_keywords]
        
        # Find sentiment words
        negative_words = [token for token in tokens if token in self.sentiment_words['negative']]
        
        # Find emotion indicators
        emotion_words = [token for token in tokens if token in self.emotion_indicators]
        
        # Detect contexts
        contexts = self._detect_context(text)
        
        # Calculate derived flags
        is_urgent = urgency_score > 0.6 or any(word in ['emergency', 'urgent', 'immediate'] for word in urgency_words)
        needs_immediate_attention = urgency_score > 0.8 or contexts.get('has_emergency_context', False)
        suggests_escalation = len(escalation_words) > 0 or urgency_score > 0.7
        
        logger.info(f"Enhanced text processed: words={word_count} chars={char_count} urgency={urgency_score:.2f}")
        
        return EnhancedTextFeatures(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            word_count=word_count,
            char_count=char_count,
            
            railway_keywords=railway_keywords,
            infrastructure_indicators=category_indicators['Infrastructure'],
            safety_indicators=category_indicators['Safety'],
            cleanliness_indicators=category_indicators['Cleanliness'],
            staff_indicators=category_indicators['Staff'],
            food_indicators=category_indicators['Food'],
            
            urgency_score=urgency_score,
            urgency_indicators=urgency_words,
            escalation_keywords=escalation_words,
            
            negative_sentiment_words=negative_words,
            emotion_indicators=emotion_words,
            
            has_emergency_context=contexts.get('has_emergency_context', False),
            has_complaint_context=contexts.get('has_complaint_context', False),
            has_suggestion_context=contexts.get('has_suggestion_context', False),
            
            is_urgent=is_urgent,
            needs_immediate_attention=needs_immediate_attention,
            suggests_escalation=suggests_escalation
        )


# Test the enhanced processor
if __name__ == "__main__":
    processor = EnhancedTextProcessor()
    
    test_texts = [
        "The overhead wire is sparking dangerously near platform 3!",
        "Toilet is extremely dirty with cockroaches everywhere",
        "TTE was very rude and demanded bribe for upper berth",
        "Food served was stale and caused food poisoning to multiple passengers",
        "Emergency! Train derailed and passengers are injured badly",
        "Suggestion: Install more CCTV cameras on platforms for better security"
    ]
    
    print("🧪 Testing Enhanced Text Processor")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        features = processor.process(text, extract_features=True)
        
        print(f"\n{i}. Text: '{text}'")
        print(f"   Railway Keywords: {features.railway_keywords}")
        print(f"   Category Indicators: I:{len(features.infrastructure_indicators)} "
              f"S:{len(features.safety_indicators)} C:{len(features.cleanliness_indicators)} "
              f"St:{len(features.staff_indicators)} F:{len(features.food_indicators)}")
        print(f"   Urgency Score: {features.urgency_score:.2f}")
        print(f"   Flags: Urgent={features.is_urgent} Immediate={features.needs_immediate_attention} "
              f"Escalate={features.suggests_escalation}")
        print(f"   Context: Emergency={features.has_emergency_context} "
              f"Complaint={features.has_complaint_context}")