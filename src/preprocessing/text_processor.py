"""
Text Processing Module for RCMS (Railway Complaint Management System).

This module handles:
- text cleaning & normalization
- tokenization
- stopword removal
- simple keyword extraction (railway-specific)
- urgency detection (heuristic)
- a lightweight `process` API that returns a ProcessedText dataclass

Dependencies: only Python stdlib + optional nltk/textblob if available.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import Counter

# Optional NLP libs (used only if available)
try:
    import nltk  # type: ignore
    from nltk import pos_tag, sent_tokenize, word_tokenize  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Exceptions
class PreprocessingError(Exception):
    """Raised when preprocessing fails."""


class RCMSException(Exception):
    """Generic RCMS exception wrapper."""


@dataclass
class ProcessedText:
    """Container for processed text data."""
    original: str
    cleaned: str
    tokens: List[str]
    word_count: int
    char_count: int
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextProcessor:
    """Advanced text processor for railway complaints.

    Responsibilities:
      - basic cleaning & normalization
      - tokenization & stopword removal
      - simple lemmatization (if NLTK available)
      - railway keyword extraction
      - urgency detection (heuristic)
      - batch processing & an ML-prep helper
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self.stop_words = self._load_stop_words()
        self.railway_keywords = self._load_railway_keywords()
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None

        # If using NLTK, ensure required data is present (best-effort)
        if NLTK_AVAILABLE:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            except Exception as e:
                logger.info("NLTK data download skipped/failed: %s", e)

    # -------------------------
    # Resources
    # -------------------------
    def _load_stop_words(self) -> set:
        """Return a set of common English stop words (simple built-in list)."""
        # Small curated list to remain Python-stdlib-only friendly
        return {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
            "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once"
        }

    def _load_railway_keywords(self) -> Dict[str, List[str]]:
        """Return railway-specific keywords organized by category."""
        return {
            "infrastructure": [
                "train", "coach", "compartment", "seat", "berth", "window",
                "door", "fan", "light", "ac", "air conditioning", "toilet",
                "platform", "station", "track", "rail"
            ],
            "cleanliness": ["dirty", "clean", "garbage", "trash", "smell", "odor", "hygiene", "sanitation"],
            "safety": ["safety", "danger", "accident", "emergency", "fire", "smoke", "thief", "robbery", "harassment"],
            "staff": ["conductor", "staff", "rude", "helpful", "service", "tte", "guard", "driver"],
            "catering": ["food", "meal", "catering", "pantry", "quality", "water", "tea", "coffee"]
        }

    # -------------------------
    # Cleaning & Tokenization
    # -------------------------
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """Clean and normalize text.

        Steps:
          - lowercasing
          - remove URLs, emails, phones
          - keep basic punctuation, strip control chars
          - optionally more aggressive cleaning (remove non-alphanumerics)
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            text = text.strip()
            # Remove URLs and emails
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
            text = re.sub(r"\S+@\S+", "", text)
            # Remove phone numbers (basic)
            text = re.sub(r"\+?\d[\d\s\-\(\)]{6,}\d", "", text)

            # Normalize whitespace and lowercase
            text = re.sub(r"\s+", " ", text).strip()
            text = text.lower()

            if aggressive:
                # Remove any non-word characters except basic punctuation
                text = re.sub(r"[^\w\s.,!?-]", " ", text)

            # Collapse multiple punctuation into a single period
            text = re.sub(r"[.,!?-]{2,}", ".", text)
            # Strip leading/trailing punctuation
            text = text.strip(".,!?- ")

            return text
        except Exception as e:
            logger.error("Error cleaning text: %s", e)
            raise PreprocessingError(f"Text cleaning failed: {e}")

    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """Tokenize text into words (simple regex-based)."""
        if not text:
            return []

        try:
            # Remove punctuation for tokenization clarity
            cleaned = re.sub(r"[^\w\s]", " ", text)
            tokens = re.findall(r"\b\w+\b", cleaned.lower())
            if remove_stop_words:
                tokens = [t for t in tokens if t not in self.stop_words]
            # remove single-char tokens and pure numbers
            tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
            return tokens
        except Exception as e:
            logger.error("Tokenization failed: %s", e)
            raise PreprocessingError(f"Tokenization failed: {e}")

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        if not NLTK_AVAILABLE or not self.lemmatizer:
            # fallback: return tokens unchanged
            return tokens
        try:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        except Exception as e:
            logger.warning("Lemmatization failed: %s", e)
            return tokens

    # -------------------------
    # Keyword extraction & NER-like helpers
    # -------------------------
    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """Return a mapping of railway categories to matched keywords found in text."""
        if not text:
            return {}
        text_lower = text.lower()
        found: Dict[str, List[str]] = {}
        for category, kws in self.railway_keywords.items():
            matches = [kw for kw in kws if kw in text_lower]
            if matches:
                found[category] = matches
        return found

    # -------------------------
    # Urgency detection (heuristic)
    # -------------------------
    def detect_urgency_indicators(self, text: str) -> Dict[str, Any]:
        """Heuristic urgency scoring based on keyword hits."""
        if not text:
            return {"urgency_score": 0, "indicators": [], "level": "low"}

        urgency_keywords = {
            "critical": ["emergency", "urgent", "critical", "immediate", "asap", "help", "danger"],
            "high": ["problem", "issue", "broken", "serious", "stuck", "injury", "fire", "smoke", "theft", "robbery"],
            "medium": ["complaint", "concern", "delay", "late", "missing"],
            "low": ["suggestion", "feedback", "minor", "small"]
        }

        text_lower = text.lower()
        score = 0
        indicators: List[str] = []
        for level, kws in urgency_keywords.items():
            for kw in kws:
                if kw in text_lower:
                    indicators.append(kw)
                    if level == "critical":
                        score += 4
                    elif level == "high":
                        score += 3
                    elif level == "medium":
                        score += 2
                    else:
                        score += 1

        level = self._score_to_level(score)
        return {"urgency_score": score, "indicators": indicators, "level": level}

    def _score_to_level(self, score: int) -> str:
        if score >= 8:
            return "critical"
        if score >= 5:
            return "high"
        if score >= 2:
            return "medium"
        return "low"

    # -------------------------
    # Named entity extraction (lightweight optional use of NLTK)
    # -------------------------
    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Return simple named entities if NLTK is available (best-effort)."""
        if not text:
            return []
        if not NLTK_AVAILABLE:
            return []
        try:
            doc_tokens = word_tokenize(text)
            tagged = pos_tag(doc_tokens)
            # This is not a full NER — just returns nouns as "entities"
            entities = []
            idx = 0
            for token, pos in tagged:
                if pos.startswith("NN"):
                    start = text.lower().find(token.lower(), idx)
                    end = start + len(token) if start >= 0 else -1
                    entities.append({"text": token, "label": "NOUN", "start": start, "end": end})
                    idx = end if end >= 0 else idx
            return entities
        except Exception as e:
            logger.warning("Named entity extraction failed: %s", e)
            return []

    # -------------------------
    # High-level process API
    # -------------------------
    def process(self, text: str, extract_features: bool = True) -> ProcessedText:
        """Complete text processing pipeline.

        Returns a ProcessedText object with metadata (keywords, urgency, basic stats).
        """
        if not text:
            raise PreprocessingError("Empty text provided")

        try:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            tokens_no_stop = self.remove_stopwords(tokens)
            lemmatized = self.lemmatize_tokens(tokens_no_stop)
            word_count = len(tokens)
            char_count = len(cleaned)

            processed = ProcessedText(
                original=text,
                cleaned=cleaned,
                tokens=lemmatized,
                word_count=word_count,
                char_count=char_count,
                language=self.language,
            )

            if extract_features:
                processed.metadata.update({
                    "keywords": self.extract_keywords(cleaned),
                    "urgency": self.detect_urgency_indicators(cleaned),
                    "has_railway_context": bool(self.extract_keywords(cleaned)),
                    "avg_word_length": (sum(len(w) for w in tokens) / len(tokens)) if tokens else 0.0
                })

            logger.info("Text processed successfully: words=%d chars=%d", word_count, char_count)
            return processed
        except Exception as e:
            logger.error("Error processing text: %s", e)
            raise PreprocessingError(f"Text processing failed: {e}")

    def batch_process(self, texts: List[str]) -> List[ProcessedText]:
        """Process multiple texts in a batch. Returns list of ProcessedText objects."""
        results: List[ProcessedText] = []
        for i, t in enumerate(texts):
            try:
                results.append(self.process(t))
            except Exception as e:
                logger.warning("Failed to process text index %d: %s", i, e)
                results.append(ProcessedText(original=t or "", cleaned="", tokens=[], word_count=0, char_count=0, metadata={"error": str(e)}))
        return results

    def preprocess_for_ml(self, text: str, include_entities: bool = True) -> Dict[str, Any]:
        """Return a dictionary of cleaned & engineered features suitable for ML input."""
        try:
            p = self.process(text, extract_features=True)
            result = {
                "original_text": p.original,
                "cleaned_text": p.cleaned,
                "tokens": p.tokens,
                "keyword_map": p.metadata.get("keywords", {}),
                "urgency": p.metadata.get("urgency", {}),
                "has_railway_context": p.metadata.get("has_railway_context", False),
                "avg_word_length": p.metadata.get("avg_word_length", 0.0),
                "word_count": p.word_count,
                "char_count": p.char_count,
                "language": p.language,
            }
            if include_entities:
                result["entities"] = self.extract_named_entities(p.cleaned)
            return result
        except Exception as e:
            logger.error("preprocess_for_ml failed: %s", e)
            raise RCMSException(f"Text preprocessing error: {e}")


# Global singleton accessor (convenience)
_preprocessor: Optional[TextProcessor] = None


def get_text_preprocessor() -> TextProcessor:
    """Return a global TextProcessor instance (lazy init)."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextProcessor()
    return _preprocessor
