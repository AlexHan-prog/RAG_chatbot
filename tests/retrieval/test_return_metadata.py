import pytest
from unittest.mock import patch, MagicMock
from langextract.core.data import AnnotatedDocument, Extraction
from src.rag_chatbot.rag.retrieval_utils import return_metadata


class TestReturnMetadata:
    """Unit tests for return_metadata function."""

    @pytest.fixture
    def mock_earnings_call_document(self):
        """Fixture for mock earnings call AnnotatedDocument"""
        return AnnotatedDocument(
            extractions=[
                Extraction(
                    extraction_class="document type",
                    extraction_text="Earnings Call",
                    attributes={"document_type": "earnings_call"},
                ),
                Extraction(
                    extraction_class="company",
                    extraction_text="Apple",
                    attributes={"company": "Apple"},
                ),
                Extraction(
                    extraction_class="year",
                    extraction_text="2024",
                    attributes={"year": "2024"},
                ),
            ]
        )

    @pytest.fixture
    def mock_meeting_notes_document(self):
        """Fixture for mock meeting notes AnnotatedDocument"""
        return AnnotatedDocument(
            extractions=[
                Extraction(
                    extraction_class="document type",
                    extraction_text="meeting notes",
                    attributes={"document_type": "meeting_note"},
                ),
                Extraction(
                    extraction_class="author",
                    extraction_text="John",
                    attributes={"author": "John"},
                ),
                Extraction(
                    extraction_class="meetingDate",
                    extraction_text="2024/03/15",
                    attributes={"meetingDate": "2024/03/15"},
                ),
            ]
        )

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_returns_annotated_document(self, mock_extract, mock_earnings_call_document):
        """Test that return_metadata returns an AnnotatedDocument."""
        mock_extract.return_value = mock_earnings_call_document
        
        result = return_metadata("Apple earnings call 2024")
        
        assert isinstance(result, AnnotatedDocument)

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_calls_extract_with_query(self, mock_extract, mock_earnings_call_document):
        """Test that return_metadata calls lx.extract with the query."""
        mock_extract.return_value = mock_earnings_call_document
        query = "Apple earnings call Q2 2024"
        
        return_metadata(query)
        
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args.kwargs["text_or_documents"] == query


    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_includes_prompt_description(self, mock_extract, mock_earnings_call_document):
        """Test that return_metadata provides a prompt description."""
        mock_extract.return_value = mock_earnings_call_document
        
        return_metadata("test query")
        
        call_args = mock_extract.call_args
        assert "prompt_description" in call_args.kwargs
        assert isinstance(call_args.kwargs["prompt_description"], str)
        assert len(call_args.kwargs["prompt_description"]) > 0

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_provides_examples(self, mock_extract, mock_earnings_call_document):
        """Test that return_metadata provides few-shot examples."""
        mock_extract.return_value = mock_earnings_call_document
        
        return_metadata("test query")
        
        call_args = mock_extract.call_args
        assert "examples" in call_args.kwargs
        examples = call_args.kwargs["examples"]
        assert len(examples) == 2  # Should have 2 examples

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_earnings_call_example(self, mock_extract, mock_earnings_call_document):
        """Test that first example is an earnings call."""
        mock_extract.return_value = mock_earnings_call_document
        
        return_metadata("test query")
        
        call_args = mock_extract.call_args
        examples = call_args.kwargs["examples"]
        first_example = examples[0]
        
        # Check that the example contains earnings call text
        assert "earnings call" in first_example.text.lower() or "agilent" in first_example.text.lower()

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_meeting_notes_example(self, mock_extract, mock_earnings_call_document):
        """Test that second example is a meeting note."""
        mock_extract.return_value = mock_earnings_call_document
        
        return_metadata("test query")
        
        call_args = mock_extract.call_args
        examples = call_args.kwargs["examples"]
        second_example = examples[1]
        
        # Check that the example contains meeting notes text
        assert "meeting notes" in second_example.text.lower() or "reuben" in second_example.text.lower()

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_earnings_call_query(self, mock_extract, mock_earnings_call_document):
        """Test return_metadata with an earnings call query."""
        mock_extract.return_value = mock_earnings_call_document
        
        result = return_metadata("What were the highlights from Apple's Q2 2024 earnings call?")
        
        assert result is not None
        assert isinstance(result, AnnotatedDocument)

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_meeting_notes_query(self, mock_extract, mock_meeting_notes_document):
        """Test return_metadata with a meeting notes query."""
        mock_extract.return_value = mock_meeting_notes_document
        
        result = return_metadata("Show me John's meeting notes from March 15th")
        
        assert result is not None
        assert isinstance(result, AnnotatedDocument)

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_empty_extractions(self, mock_extract):
        """Test return_metadata with no extractions."""
        empty_doc = AnnotatedDocument(extractions=[])
        mock_extract.return_value = empty_doc
        
        result = return_metadata("random query")
        
        assert result is not None
        assert len(result.extractions) == 0

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_preserves_extraction_details(self, mock_extract, mock_earnings_call_document):
        """Test that return_metadata preserves extraction attributes."""
        mock_extract.return_value = mock_earnings_call_document
        
        result = return_metadata("test query")
        
        # Check that extractions have attributes
        for extraction in result.extractions:
            assert extraction.attributes is not None
            assert isinstance(extraction.attributes, dict)

    @patch("src.rag_chatbot.rag.retrieval_utils.lx.extract")
    def test_return_metadata_handles_multiple_extractions(self, mock_extract):
        """Test return_metadata with multiple extractions."""
        doc = AnnotatedDocument(
            extractions=[
                Extraction(extraction_class="company", extraction_text="Apple", attributes={"company": "Apple"}),
                Extraction(extraction_class="company", extraction_text="Agilent", attributes={"company": "Agilent"}),
                Extraction(extraction_class="year", extraction_text="2024", attributes={"year": "2024"}),
                Extraction(extraction_class="quarter", extraction_text="Q2", attributes={"quarter": "2"}),
            ]
        )
        mock_extract.return_value = doc
        
        result = return_metadata("Apple and Agilent in Q2 2024")
        
        assert len(result.extractions) == 4
