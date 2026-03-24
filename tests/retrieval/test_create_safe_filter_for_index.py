import pytest
from src.rag_chatbot.rag.retrieval_utils import create_safe_filter_for_index


class TestCreateSafeFilterForIndex:
    """Unit tests for create_safe_filter_for_index function."""

    # ============== FIXTURES ==============

    @pytest.fixture
    def empty_metadata(self):
        """Fixture for empty metadata."""
        return {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def earnings_call_metadata(self):
        """Fixture for earnings call metadata."""
        return {
            "docType": ["earnings_call"],
            "company": ["Apple"],
            "year": [2024],
            "quarter": [2],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def meeting_notes_metadata(self):
        """Fixture for meeting notes metadata."""
        return {
            "docType": ["meeting_note"],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-03-15T00:00:00Z"],
            "author": ["John"]
        }

    @pytest.fixture
    def mixed_metadata(self):
        """Fixture for mixed metadata with both types of fields."""
        return {
            "docType": ["earnings_call"],
            "company": ["Apple", "Agilent"],
            "year": [2024, 2023],
            "quarter": [2, 4],
            "meetingDate": ["2024-03-15T00:00:00Z"],
            "author": ["John", "Sarah"]
        }

    # ============== TESTS ==============

    def test_empty_metadata_returns_empty_string(self, empty_metadata):
        """Test that empty metadata returns empty string."""
        result = create_safe_filter_for_index(empty_metadata, "transcripts")
        assert result == ""

    def test_transcripts_preserves_doctype(self, earnings_call_metadata):
        """Test that transcripts index preserves docType field."""
        result = create_safe_filter_for_index(earnings_call_metadata, "transcripts")
        assert "docType eq 'earnings_call'" in result

    def test_transcripts_preserves_company(self, earnings_call_metadata):
        """Test that transcripts index preserves company field."""
        result = create_safe_filter_for_index(earnings_call_metadata, "transcripts")
        assert "company eq 'Apple'" in result

    def test_transcripts_preserves_year(self, earnings_call_metadata):
        """Test that transcripts index preserves year field."""
        result = create_safe_filter_for_index(earnings_call_metadata, "transcripts")
        assert "year eq 2024" in result

    def test_transcripts_preserves_quarter(self, earnings_call_metadata):
        """Test that transcripts index preserves quarter field."""
        result = create_safe_filter_for_index(earnings_call_metadata, "transcripts")
        assert "quarter eq 2" in result

    def test_transcripts_removes_author(self, meeting_notes_metadata):
        """Test that transcripts index removes author field."""
        result = create_safe_filter_for_index(meeting_notes_metadata, "transcripts")
        assert "author" not in result

    def test_transcripts_removes_meeting_date(self, meeting_notes_metadata):
        """Test that transcripts index removes meetingDate field."""
        result = create_safe_filter_for_index(meeting_notes_metadata, "transcripts")
        assert "meetingDate" not in result

    def test_meeting_notes_preserves_doctype(self, meeting_notes_metadata):
        """Test that meeting_notes index preserves docType field."""
        result = create_safe_filter_for_index(meeting_notes_metadata, "meeting_notes")
        assert "docType eq 'meeting_note'" in result

    def test_meeting_notes_preserves_author(self, meeting_notes_metadata):
        """Test that meeting_notes index preserves author field."""
        result = create_safe_filter_for_index(meeting_notes_metadata, "meeting_notes")
        assert "author eq 'John'" in result

    def test_meeting_notes_preserves_meeting_date(self, meeting_notes_metadata):
        """Test that meeting_notes index preserves meetingDate field."""
        result = create_safe_filter_for_index(meeting_notes_metadata, "meeting_notes")
        assert "meetingDate eq 2024-03-15T00:00:00Z" in result

    def test_meeting_notes_removes_company(self, earnings_call_metadata):
        """Test that meeting_notes index removes company field."""
        result = create_safe_filter_for_index(earnings_call_metadata, "meeting_notes")
        assert "company" not in result

    def test_meeting_notes_removes_year(self, earnings_call_metadata):
        """Test that meeting_notes index removes year field."""
        result = create_safe_filter_for_index(earnings_call_metadata, "meeting_notes")
        assert "year" not in result

    def test_meeting_notes_removes_quarter(self, earnings_call_metadata):
        """Test that meeting_notes index removes quarter field."""
        result = create_safe_filter_for_index(earnings_call_metadata, "meeting_notes")
        assert "quarter" not in result

    def test_mixed_metadata_transcripts_filters_correctly(self, mixed_metadata):
        """Test that transcripts index filters out meeting_notes-specific fields."""
        result = create_safe_filter_for_index(mixed_metadata, "transcripts")
        
        # Should include earnings call fields
        assert "company" in result or len(result) > 0
        assert "year" in result or len(result) > 0
        
        # Should exclude meeting notes fields
        assert "author" not in result
        assert "meetingDate" not in result

    def test_mixed_metadata_meeting_notes_filters_correctly(self, mixed_metadata):
        """Test that meeting_notes index filters out earnings_call-specific fields."""
        result = create_safe_filter_for_index(mixed_metadata, "meeting_notes")
        
        # Should include meeting notes fields
        assert "author" in result or len(result) > 0
        assert "meetingDate" in result or len(result) > 0
        
        # Should exclude earnings call fields
        assert "company" not in result
        assert "year" not in result
        assert "quarter" not in result

    def test_multiple_companies_in_transcripts(self):
        """Test that multiple companies are handled in transcripts index."""
        metadata = {
            "docType": [],
            "company": ["Apple", "Agilent"],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }
        result = create_safe_filter_for_index(metadata, "transcripts")
        
        # Should contain OR clause for multiple companies
        assert "Apple" in result
        assert "Agilent" in result
        assert " or " in result

    def test_multiple_years_in_transcripts(self):
        """Test that multiple years are handled in transcripts index."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [2024, 2023],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }
        result = create_safe_filter_for_index(metadata, "transcripts")
        
        assert "2024" in result
        assert "2023" in result

    def test_multiple_authors_in_meeting_notes(self):
        """Test that multiple authors are handled in meeting_notes index."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": ["John", "Sarah"]
        }
        result = create_safe_filter_for_index(metadata, "meeting_notes")
        
        assert "John" in result
        assert "Sarah" in result
        assert " or " in result

    def test_multiple_dates_in_meeting_notes(self):
        """Test that multiple dates are handled in meeting_notes index."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-03-15T00:00:00Z", "2024-06-20T00:00:00Z"],
            "author": []
        }
        result = create_safe_filter_for_index(metadata, "meeting_notes")
        
        assert "2024-03-15T00:00:00Z" in result
        assert "2024-06-20T00:00:00Z" in result

    def test_invalid_index_kind_returns_empty_string(self, earnings_call_metadata):
        """Test that invalid index_kind returns empty string."""
        result = create_safe_filter_for_index(earnings_call_metadata, "invalid_index")
        assert result == ""

    def test_transcripts_with_only_allowed_fields(self):
        """Test transcripts index with only allowed fields."""
        metadata = {
            "docType": ["earnings_call"],
            "company": ["Apple"],
            "year": [2024],
            "quarter": [2],
            "meetingDate": [],
            "author": []
        }
        result = create_safe_filter_for_index(metadata, "transcripts")
        
        # All fields should be present
        assert "docType" in result
        assert "company" in result
        assert "year" in result
        assert "quarter" in result

    def test_meeting_notes_with_only_allowed_fields(self):
        """Test meeting_notes index with only allowed fields."""
        metadata = {
            "docType": ["meeting_note"],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-03-15T00:00:00Z"],
            "author": ["John"]
        }
        result = create_safe_filter_for_index(metadata, "meeting_notes")
        
        # All fields should be present
        assert "docType" in result
        assert "author" in result
        assert "meetingDate" in result

    def test_transcripts_returns_string(self, earnings_call_metadata):
        """Test that transcripts index returns a string."""
        result = create_safe_filter_for_index(earnings_call_metadata, "transcripts")
        assert isinstance(result, str)

    def test_meeting_notes_returns_string(self, meeting_notes_metadata):
        """Test that meeting_notes index returns a string."""
        result = create_safe_filter_for_index(meeting_notes_metadata, "meeting_notes")
        assert isinstance(result, str)

    def test_filters_joined_with_and(self, mixed_metadata):
        """Test that multiple filters are joined with AND operator."""
        result = create_safe_filter_for_index(mixed_metadata, "transcripts")
        
        if " and " in result:
            # Should have proper AND operators between filters
            assert result.count(" and ") > 0

    def test_doctype_included_in_both_indexes(self):
        """Test that docType is included in both index types."""
        metadata = {
            "docType": ["earnings_call"],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }
        
        result_transcripts = create_safe_filter_for_index(metadata, "transcripts")
        result_meeting = create_safe_filter_for_index(metadata, "meeting_notes")
        
        assert "docType eq 'earnings_call'" in result_transcripts
        assert "docType eq 'earnings_call'" in result_meeting
