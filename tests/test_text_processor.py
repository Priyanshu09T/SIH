import pytestimport pytest"""""""""

import sys

from pathlib import Pathimport sys



# Add src to Python pathfrom pathlib import PathTest cases for TextProcessor module.

src_path = Path(__file__).parent.parent / "src"

sys.path.insert(0, str(src_path))



from src.preprocessing.text_processor import TextProcessor, ProcessedText# Add src to Python path"""Test cases for TextProcessor module.Test module for text preprocessing functionality.



src_path = Path(__file__).parent.parent / "src"

class TestTextProcessor:

    sys.path.insert(0, str(src_path))

    @pytest.fixture

    def processor(self):

        return TextProcessor()

    from src.preprocessing.text_processor import TextProcessor, ProcessedTextimport pytestComprehensive testing of text processing functionality."""

    def test_processor_initialization(self, processor):

        assert processor is not None

        assert isinstance(processor.stop_words, set)

        assert isinstance(processor.railway_keywords, dict)import sys

    

    def test_clean_text_basic(self, processor):class TestTextProcessor:

        text = "The TRAIN is very DIRTY!"

        cleaned = processor.clean_text(text)    from pathlib import Path"""

        assert cleaned == "the train is very dirty"

            @pytest.fixture

        assert processor.clean_text("") == ""

        assert processor.clean_text(None) == ""    def processor(self):

    

    def test_tokenize_basic(self, processor):        return TextProcessor()

        text = "The train is dirty"

        tokens = processor.tokenize(text, remove_stop_words=False)    # Add src to Python pathimport pytest

        

        assert isinstance(tokens, list)    def test_processor_initialization(self, processor):

        assert len(tokens) > 0

        assert "train" in tokens        assert processor is not Nonesrc_path = Path(__file__).parent.parent / "src"

    

    def test_extract_keywords(self, processor):        assert isinstance(processor.stop_words, set)

        text = "The train coach is dirty"

        keywords = processor.extract_keywords(text)        assert isinstance(processor.railway_keywords, dict)sys.path.insert(0, str(src_path))import pytestfrom src.preprocessing.text_processor import TextPreprocessor, get_text_preprocessor

        

        assert isinstance(keywords, dict)    

    

    def test_process_pipeline(self, processor):    def test_clean_text_basic(self, processor):

        text = "The train toilet is dirty!"

        processed = processor.process(text, extract_features=True)        text = "The TRAIN is very DIRTY!"

        

        assert isinstance(processed, ProcessedText)        cleaned = processor.clean_text(text)from src.preprocessing.text_processor import TextProcessor, ProcessedTextimport sys

        assert processed.original == text

        assert len(processed.cleaned) > 0        assert cleaned == "the train is very dirty"

        assert len(processed.tokens) > 0

        assert processed.word_count > 0        from src.utils.exceptions import PreprocessingError

        assert processed.char_count > 0
        assert processor.clean_text("") == ""

        assert processor.clean_text(None) == ""from pathlib import Path

    

    def test_tokenize_basic(self, processor):

        text = "The train is dirty"

        tokens = processor.tokenize(text, remove_stop_words=False)class TestTextProcessor:class TestTextPreprocessor:

        

        assert isinstance(tokens, list)    """Test cases for TextProcessor class."""

        assert len(tokens) > 0

        assert "train" in tokens    # Add src to Python path    """Test cases for TextPreprocessor class."""

    

    def test_extract_keywords(self, processor):    @pytest.fixture

        text = "The train coach is dirty"

        keywords = processor.extract_keywords(text)    def processor(self):src_path = Path(__file__).parent.parent / "src"    

        

        assert isinstance(keywords, dict)        """Create TextProcessor instance for testing."""

    

    def test_process_pipeline(self, processor):        return TextProcessor()sys.path.insert(0, str(src_path))    def setup_method(self):

        text = "The train toilet is dirty!"

        processed = processor.process(text, extract_features=True)    

        

        assert isinstance(processed, ProcessedText)    def test_processor_initialization(self, processor):        """Setup test fixtures."""

        assert processed.original == text

        assert len(processed.cleaned) > 0        """Test processor initializes correctly."""

        assert len(processed.tokens) > 0

        assert processed.word_count > 0        assert processor is not Nonefrom src.preprocessing.text_processor import TextProcessor, ProcessedText        self.preprocessor = TextPreprocessor()

        assert processed.char_count > 0
        assert isinstance(processor.stop_words, set)

        assert isinstance(processor.railway_keywords, dict)from src.utils.exceptions import PreprocessingError        self.sample_text = "The train is very dirty and the toilet smells horrible!"

        assert len(processor.stop_words) > 0

        assert len(processor.railway_keywords) > 0        self.complex_text = """

    

    def test_clean_text_basic(self, processor):        I am writing to complain about the dirty condition of the train.

        """Test basic text cleaning functionality."""

        # Test normal textclass TestTextProcessor:        The coach number 12345 was extremely unclean. There was garbage everywhere

        text = "The TRAIN is very DIRTY and smells bad!"

        cleaned = processor.clean_text(text)    """Test cases for TextProcessor class."""        and the toilet was not working properly. This is unacceptable!

        assert cleaned == "the train is very dirty and smells bad"

                    Contact me at john.doe@email.com or call +91-9876543210.

        # Test empty text

        assert processor.clean_text("") == ""    @pytest.fixture        """

        assert processor.clean_text(None) == ""

        def processor(self):    

    def test_tokenize_basic(self, processor):

        """Test basic tokenization."""        """Create TextProcessor instance for testing."""    def test_clean_text_basic(self):

        text = "The train is very dirty and smells bad"

        tokens = processor.tokenize(text, remove_stop_words=False)        return TextProcessor()        """Test basic text cleaning."""

        

        assert isinstance(tokens, list)            result = self.preprocessor.clean_text(self.sample_text)

        assert len(tokens) > 0

        assert "train" in tokens    def test_processor_initialization(self, processor):        

        assert "dirty" in tokens

            """Test processor initializes correctly."""        assert isinstance(result, str)

    def test_extract_keywords(self, processor):

        """Test railway keyword extraction."""        assert processor is not None        assert len(result) > 0

        text = "The train coach is dirty and the toilet smells bad"

        keywords = processor.extract_keywords(text)        assert isinstance(processor.stop_words, set)        assert result.islower()

        

        assert isinstance(keywords, dict)        assert isinstance(processor.railway_keywords, dict)    

        # Should find infrastructure and cleanliness keywords

        expected_categories = ["infrastructure", "cleanliness"]        assert len(processor.stop_words) > 0    def test_clean_text_aggressive(self):

        found_categories = list(keywords.keys())

                assert len(processor.railway_keywords) > 0        """Test aggressive text cleaning."""

        # At least one category should be found

        assert len(found_categories) > 0            result = self.preprocessor.clean_text(self.complex_text, aggressive=True)

    

    def test_detect_urgency(self, processor):    def test_clean_text_basic(self, processor):        

        """Test urgency detection."""

        # Test critical urgency        """Test basic text cleaning functionality."""        # Should remove email and phone number

        critical_text = "Emergency! This is urgent!"

        urgency = processor.detect_urgency_indicators(critical_text)        # Test normal text        assert "john.doe@email.com" not in result

        

        assert urgency["level"] in ["critical", "high"]        text = "The TRAIN is very DIRTY and smells bad!"        assert "+91-9876543210" not in result

        assert urgency["urgency_score"] > 0

            cleaned = processor.clean_text(text)        assert "9876543210" not in result

    def test_process_pipeline(self, processor):

        """Test complete processing pipeline."""        assert cleaned == "the train is very dirty and smells bad"    

        text = "The train toilet is dirty! This is urgent!"

                    def test_tokenize(self):

        processed = processor.process(text, extract_features=True)

                # Test empty text        """Test tokenization."""

        # Check ProcessedText structure

        assert isinstance(processed, ProcessedText)        assert processor.clean_text("") == ""        tokens = self.preprocessor.tokenize(self.sample_text)

        assert processed.original == text

        assert len(processed.cleaned) > 0        assert processor.clean_text(None) == ""        

        assert len(processed.tokens) > 0

        assert processed.word_count > 0                assert isinstance(tokens, list)

        assert processed.char_count > 0

                # Test text with extra whitespace        assert len(tokens) > 0

        # Check metadata exists

        assert processed.metadata is not None        text = "  Multiple   spaces   here  "        assert "train" in [token.lower() for token in tokens]

        assert "keywords" in processed.metadata

        assert "urgency" in processed.metadata        cleaned = processor.clean_text(text)    



        assert cleaned == "multiple spaces here"    def test_remove_stopwords(self):

if __name__ == "__main__":

    pytest.main([__file__, "-v"])            """Test stopword removal."""

    def test_tokenize_basic(self, processor):        tokens = ["the", "train", "is", "very", "dirty"]

        """Test basic tokenization."""        filtered = self.preprocessor.remove_stopwords(tokens)

        text = "The train is very dirty and smells bad"        

        tokens = processor.tokenize(text, remove_stop_words=False)        assert "the" not in filtered

                assert "is" not in filtered

        assert isinstance(tokens, list)        assert "train" in filtered

        assert len(tokens) > 0        assert "dirty" in filtered

        assert "train" in tokens    

        assert "dirty" in tokens    def test_stem_tokens(self):

            """Test stemming."""

    def test_extract_keywords_railway_context(self, processor):        tokens = ["running", "runs", "ran"]

        """Test extraction of railway-specific keywords."""        stemmed = self.preprocessor.stem_tokens(tokens)

        text = "The train coach is dirty and the toilet smells bad"        

        keywords = processor.extract_keywords(text)        assert isinstance(stemmed, list)

                assert len(stemmed) == len(tokens)

        assert isinstance(keywords, dict)    

        assert "infrastructure" in keywords    def test_lemmatize_tokens(self):

        assert "cleanliness" in keywords        """Test lemmatization."""

        assert "train" in keywords["infrastructure"]        tokens = ["running", "runs", "ran"]

        assert "coach" in keywords["infrastructure"]        lemmatized = self.preprocessor.lemmatize_tokens(tokens)

        assert "dirty" in keywords["cleanliness"]        

            assert isinstance(lemmatized, list)

    def test_detect_urgency_indicators(self, processor):        assert len(lemmatized) == len(tokens)

        """Test urgency detection functionality."""    

        # Test critical urgency    def test_extract_keywords(self):

        critical_text = "Emergency! Train accident happened immediately!"        """Test keyword extraction."""

        urgency = processor.detect_urgency_indicators(critical_text)        keywords = self.preprocessor.extract_keywords(self.complex_text)

                

        assert urgency["level"] == "critical"        assert isinstance(keywords, list)

        assert urgency["urgency_score"] >= 8        assert len(keywords) > 0

        assert "emergency" in urgency["indicators"]        # Should contain relevant complaint keywords

            keyword_text = " ".join(keywords).lower()

    def test_process_complete_pipeline(self, processor):        assert any(word in keyword_text for word in ["train", "dirty", "toilet", "coach"])

        """Test complete processing pipeline."""    

        text = "The train toilet is very dirty and smells terrible! This is urgent!"    def test_detect_language(self):

                """Test language detection."""

        processed = processor.process(text, extract_features=True)        language = self.preprocessor.detect_language(self.sample_text)

                

        # Check ProcessedText structure        assert isinstance(language, str)

        assert isinstance(processed, ProcessedText)        assert len(language) == 2  # Language code should be 2 characters

        assert processed.original == text    

        assert len(processed.cleaned) > 0    def test_get_text_statistics(self):

        assert len(processed.tokens) > 0        """Test text statistics."""

        assert processed.word_count > 0        stats = self.preprocessor.get_text_statistics(self.sample_text)

        assert processed.char_count > 0        

                required_keys = [

        # Check metadata            "character_count", "word_count", "sentence_count",

        assert processed.metadata is not None            "avg_sentence_length", "avg_word_length", "unique_words", "language"

        assert "keywords" in processed.metadata        ]

        assert "urgency" in processed.metadata        

        assert "has_railway_context" in processed.metadata        for key in required_keys:

            assert key in stats

            assert isinstance(stats[key], (int, float, str))

if __name__ == "__main__":        

    # Run tests if script is executed directly        assert stats["word_count"] > 0

    pytest.main([__file__, "-v"])        assert stats["character_count"] > 0
    
    def test_preprocess_for_ml(self):
        """Test complete ML preprocessing pipeline."""
        result = self.preprocessor.preprocess_for_ml(self.sample_text)
        
        required_keys = [
            "original_text", "cleaned_text", "tokens", "tokens_no_stopwords",
            "lemmatized_tokens", "processed_text", "keywords", "statistics"
        ]
        
        for key in required_keys:
            assert key in result
        
        assert result["original_text"] == self.sample_text
        assert isinstance(result["tokens"], list)
        assert isinstance(result["keywords"], list)
        assert isinstance(result["statistics"], dict)
    
    def test_preprocess_empty_text(self):
        """Test preprocessing with empty text."""
        result = self.preprocessor.clean_text("")
        assert result == ""
        
        result = self.preprocessor.preprocess_for_ml("")
        assert result["original_text"] == ""
        assert result["cleaned_text"] == ""
    
    def test_preprocess_none_text(self):
        """Test preprocessing with None text."""
        result = self.preprocessor.clean_text(None)
        assert result == ""
    
    def test_get_global_preprocessor(self):
        """Test global preprocessor instance."""
        preprocessor1 = get_text_preprocessor()
        preprocessor2 = get_text_preprocessor()
        
        # Should return the same instance
        assert preprocessor1 is preprocessor2
        assert isinstance(preprocessor1, TextPreprocessor)


# Railway-specific test cases
class TestRailwaySpecificProcessing:
    """Test cases for railway complaint specific processing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_infrastructure_complaint(self):
        """Test processing of infrastructure complaint."""
        text = "The seat is broken and window glass is cracked in coach S4."
        result = self.preprocessor.preprocess_for_ml(text)
        
        keywords = [kw.lower() for kw in result["keywords"]]
        assert any(word in keywords for word in ["seat", "broken", "window", "coach"])
    
    def test_cleanliness_complaint(self):
        """Test processing of cleanliness complaint."""
        text = "The toilet is very dirty and there is garbage on the floor."
        result = self.preprocessor.preprocess_for_ml(text)
        
        keywords = [kw.lower() for kw in result["keywords"]]
        assert any(word in keywords for word in ["toilet", "dirty", "garbage", "floor"])
    
    def test_staff_complaint(self):
        """Test processing of staff behavior complaint."""
        text = "The ticket collector was very rude and unhelpful to passengers."
        result = self.preprocessor.preprocess_for_ml(text)
        
        keywords = [kw.lower() for kw in result["keywords"]]
        assert any(word in keywords for word in ["ticket", "collector", "rude", "passengers"])
    
    def test_mixed_language_content(self):
        """Test processing with mixed language content."""
        # This would be enhanced with proper multilingual support
        text = "Train bahut gandi hai and toilet is not working."
        result = self.preprocessor.preprocess_for_ml(text)
        
        # Should still extract some meaningful keywords
        assert len(result["keywords"]) > 0
        assert result["statistics"]["word_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__])