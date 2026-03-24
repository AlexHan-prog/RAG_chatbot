

import pytest
from datetime import datetime
from langextract.core.data import AnnotatedDocument, Extraction
from src.rag_chatbot.rag.retrieval_utils import langextract_to_metadata


class TestLangextractToMetadata:
    """Unit tests for langextract_to_metadata function."""

    def test_empty_document(self):
        """Test with an empty AnnotatedDocument has no extractions."""
        doc = AnnotatedDocument(extractions=[])
        result = langextract_to_metadata(doc)
        
        assert result["docType"] == []
        assert result["company"] == []
        assert result["year"] == []
        assert result["quarter"] == []
        assert result["meetingDate"] == []
        assert result["author"] == []

    def test_single_earnings_call_extraction(self):
        """Test extraction of single earnings call fields."""
        extractions = [
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
            Extraction(
                extraction_class="quarter",
                extraction_text="Q2",
                attributes={"quarter": "2"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert result["docType"] == ["earnings_call"]
        assert result["company"] == ["Apple"]
        assert result["year"] == [2024]
        assert result["quarter"] == [2]
        assert result["meetingDate"] == []
        assert result["author"] == []

    def test_single_meeting_note_extraction(self):
        """Test extraction of single meeting note fields."""
        extractions = [
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
                extraction_text="2024/01/15",
                attributes={"meetingDate": "2024/01/15"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert result["docType"] == ["meeting_note"]
        assert result["author"] == ["John"]
        assert result["meetingDate"] == ["2024-01-15T00:00:00Z"]
        assert result["company"] == []
        assert result["year"] == []

    def test_multiple_companies_extraction(self):
        """Test extraction of multiple company names."""
        extractions = [
            Extraction(
                extraction_class="company",
                extraction_text="Apple",
                attributes={"company": "Apple"},
            ),
            Extraction(
                extraction_class="company",
                extraction_text="Agilent",
                attributes={"company": "Agilent"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert set(result["company"]) == {"Apple", "Agilent"}
        assert len(result["company"]) == 2

    def test_multiple_years_and_quarters(self):
        """Test extraction of multiple years and quarters."""
        extractions = [
            Extraction(
                extraction_class="year",
                extraction_text="2024",
                attributes={"year": "2024"},
            ),
            Extraction(
                extraction_class="year",
                extraction_text="2023",
                attributes={"year": "2023"},
            ),
            Extraction(
                extraction_class="quarter",
                extraction_text="Q2",
                attributes={"quarter": "2"},
            ),
            Extraction(
                extraction_class="quarter",
                extraction_text="Q4",
                attributes={"quarter": "4"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert set(result["year"]) == {2024, 2023}
        assert set(result["quarter"]) == {2, 4}

    def test_duplicate_deduplication(self):
        """Test that duplicate values are removed."""
        extractions = [
            Extraction(
                extraction_class="company",
                extraction_text="Apple",
                attributes={"company": "Apple"},
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
            Extraction(
                extraction_class="year",
                extraction_text="2024",
                attributes={"year": "2024"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert result["company"] == ["Apple"]
        assert result["year"] == [2024]

    def test_mixed_fields_earnings_and_meeting(self):
        """Test extraction with mixed earnings call and meeting note fields."""
        extractions = [
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
            Extraction(
                extraction_class="author",
                extraction_text="Jane",
                attributes={"author": "Jane"},
            ),
            Extraction(
                extraction_class="meetingDate",
                extraction_text="2024/03/15",
                attributes={"meetingDate": "2024/03/15"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        # Both types of fields should be captured
        assert result["company"] == ["Apple"]
        assert result["year"] == [2024]
        assert result["author"] == ["Jane"]
        assert result["meetingDate"] == ["2024-03-15T00:00:00Z"]

    def test_quarter_string_to_int_conversion(self):
        """Test that quarter string values are converted to integers."""
        extractions = [
            Extraction(
                extraction_class="quarter",
                extraction_text="Q1",
                attributes={"quarter": "1"},
            ),
            Extraction(
                extraction_class="quarter",
                extraction_text="Q3",
                attributes={"quarter": "3"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert all(isinstance(q, int) for q in result["quarter"])
        assert set(result["quarter"]) == {1, 3}

    def test_year_string_to_int_conversion(self):
        """Test that year string values are converted to integers."""
        extractions = [
            Extraction(
                extraction_class="year",
                extraction_text="2024",
                attributes={"year": "2024"},
            ),
            Extraction(
                extraction_class="year",
                extraction_text="2025",
                attributes={"year": "2025"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert all(isinstance(y, int) for y in result["year"])
        assert set(result["year"]) == {2024, 2025}

    def test_meeting_date_iso_format_conversion(self):
        """Test that meeting dates are converted to ISO format with Z suffix."""
        extractions = [
            Extraction(
                extraction_class="meetingDate",
                extraction_text="2024/01/15",
                attributes={"meetingDate": "2024/01/15"},
            ),
            Extraction(
                extraction_class="meetingDate",
                extraction_text="2024/12/31",
                attributes={"meetingDate": "2024/12/31"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert "2024-01-15T00:00:00Z" in result["meetingDate"]
        assert "2024-12-31T00:00:00Z" in result["meetingDate"]

    def test_extraction_with_none_attributes(self):
        """Test handling of extractions with None attributes."""
        extractions = [
            Extraction(
                extraction_class="company",
                extraction_text="Apple",
                attributes=None,  # None attributes
            ),
            Extraction(
                extraction_class="company",
                extraction_text="Agilent",
                attributes={"company": "Agilent"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        # Should only capture the extraction with valid attributes
        assert result["company"] == ["Agilent"]

    def test_invalid_year_format_ignored(self):
        """Test that invalid year format is ignored (conversion fails silently)."""
        extractions = [
            Extraction(
                extraction_class="year",
                extraction_text="invalid",
                attributes={"year": "not_a_year"},
            ),
            Extraction(
                extraction_class="year",
                extraction_text="2024",
                attributes={"year": "2024"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        # Invalid year should be ignored
        assert result["year"] == [2024]

    def test_invalid_quarter_format_ignored(self):
        """Test that invalid quarter format is ignored (conversion fails silently)."""
        extractions = [
            Extraction(
                extraction_class="quarter",
                extraction_text="invalid",
                attributes={"quarter": "Q99"},
            ),
            Extraction(
                extraction_class="quarter",
                extraction_text="Q2",
                attributes={"quarter": "2"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        # Invalid quarter should be ignored
        assert result["quarter"] == [2]

    def test_invalid_date_format_raises_error(self):
        """Test that invalid date format raises an error."""
        extractions = [
            Extraction(
                extraction_class="meetingDate",
                extraction_text="invalid date",
                attributes={"meetingDate": "01-15-2024"},  # Wrong format
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        
        with pytest.raises(ValueError):
            langextract_to_metadata(doc)

    def test_all_document_types(self):
        """Test extraction of all possible document types."""
        extractions = [
            Extraction(
                extraction_class="document type",
                extraction_text="Earnings Call",
                attributes={"document_type": "earnings_call"},
            ),
            Extraction(
                extraction_class="document type",
                extraction_text="Meeting Notes",
                attributes={"document_type": "meeting_note"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert set(result["docType"]) == {"earnings_call", "meeting_note"}

    def test_comprehensive_earnings_call_metadata(self):
        """Comprehensive test with full earnings call metadata"""
        extractions = [
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
                extraction_class="company",
                extraction_text="Agilent",
                attributes={"company": "Agilent"},
            ),
            Extraction(
                extraction_class="year",
                extraction_text="2024",
                attributes={"year": "2024"},
            ),
            Extraction(
                extraction_class="year",
                extraction_text="2023",
                attributes={"year": "2023"},
            ),
            Extraction(
                extraction_class="quarter",
                extraction_text="Q2",
                attributes={"quarter": "2"},
            ),
            Extraction(
                extraction_class="quarter",
                extraction_text="Q4",
                attributes={"quarter": "4"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert result["docType"] == ["earnings_call"]
        assert set(result["company"]) == {"Apple", "Agilent"}
        assert set(result["year"]) == {2024, 2023}
        assert set(result["quarter"]) == {2, 4}
        assert result["meetingDate"] == []
        assert result["author"] == []

    def test_comprehensive_meeting_notes_metadata(self):
        """Comprehensive test with full meeting notes metadata"""
        extractions = [
            Extraction(
                extraction_class="document type",
                extraction_text="Meeting Notes",
                attributes={"document_type": "meeting_note"},
            ),
            Extraction(
                extraction_class="author",
                extraction_text="John Doe",
                attributes={"author": "John Doe"},
            ),
            Extraction(
                extraction_class="meetingDate",
                extraction_text="2024/03/15",
                attributes={"meetingDate": "2024/03/15"},
            ),
            Extraction(
                extraction_class="meetingDate",
                extraction_text="2024/06/20",
                attributes={"meetingDate": "2024/06/20"},
            ),
        ]
        doc = AnnotatedDocument(extractions=extractions)
        result = langextract_to_metadata(doc)
        
        assert result["docType"] == ["meeting_note"]
        assert result["author"] == ["John Doe"]
        assert set(result["meetingDate"]) == {"2024-03-15T00:00:00Z", "2024-06-20T00:00:00Z"}
        assert result["company"] == []
        assert result["year"] == []
        assert result["quarter"] == []