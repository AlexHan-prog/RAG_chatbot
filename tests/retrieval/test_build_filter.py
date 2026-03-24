import pytest
from src.rag_chatbot.rag.retrieval_utils import build_filter


class TestBuildFilter:
    """Unit tests for build_filter function."""

    # ============== FIXTURES ==============

    @pytest.fixture
    def empty_metadata(self):
        """Fixture for empty metadata"""
        return {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def single_earnings_metadata(self):
        """Fixture for single earnings call metadata"""
        return {
            "docType": ["earnings_call"],
            "company": ["Apple"],
            "year": [2024],
            "quarter": [2],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def single_meeting_metadata(self):
        """Fixture for single meeting note metadata"""
        return {
            "docType": ["meeting_note"],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-03-15T00:00:00Z"],
            "author": ["John"]
        }

    @pytest.fixture
    def multiple_companies_metadata(self):
        """Fixture for multiple companies"""
        return {
            "docType": [],
            "company": ["Apple", "Agilent"],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def multiple_years_metadata(self):
        """Fixture for multiple years"""
        return {
            "docType": [],
            "company": [],
            "year": [2024, 2023],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def multiple_quarters_metadata(self):
        """Fixture for multiple quarters"""
        return {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [1, 3, 4],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def multiple_dates_metadata(self):
        """Fixture for multiple meeting dates"""
        return {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-01-15T00:00:00Z", "2024-06-20T00:00:00Z"],
            "author": []
        }

    @pytest.fixture
    def complex_metadata(self):
        """Fixture for complex metadata with multiple fields populated"""
        return {
            "docType": ["earnings_call"],
            "company": ["Apple", "Agilent"],
            "year": [2024, 2023],
            "quarter": [2, 4],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def mixed_metadata(self):
        """Fixture for mixed earnings call and meeting note metadata"""
        return {
            "docType": ["earnings_call"],
            "company": ["Microsoft"],
            "year": [2024],
            "quarter": [3],
            "meetingDate": ["2024-07-15T00:00:00Z"],
            "author": ["Sarah"]
        }

    # ============== TESTS ==============

    def test_empty_metadata(self, empty_metadata):
        """Test with completely empty metadata."""
        result = build_filter(empty_metadata)
        assert result == ""

    def test_single_doctype(self):
        """Test with only docType field."""
        metadata = {
            "docType": ["earnings_call"],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }
        result = build_filter(metadata)
        assert result == "docType eq 'earnings_call'"

    def test_single_company(self):
        """Test with only a single company."""
        metadata = {
            "docType": [],
            "company": ["Apple"],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }
        result = build_filter(metadata)
        assert result == "company eq 'Apple'"

    def test_single_year(self):
        """Test with only a single year."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [2024],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }
        result = build_filter(metadata)
        assert result == "year eq 2024"

    def test_single_quarter(self):
        """Test with only a single quarter."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [3],
            "meetingDate": [],
            "author": []
        }
        result = build_filter(metadata)
        assert result == "quarter eq 3"

    def test_single_author(self):
        """Test with only a single author."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": ["John"]
        }
        result = build_filter(metadata)
        assert result == "author eq 'John'"

    def test_single_date(self):
        """Test with only a single meeting date."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-03-15T00:00:00Z"],
            "author": []
        }
        result = build_filter(metadata)
        assert result == "meetingDate eq 2024-03-15T00:00:00Z"

    def test_multiple_companies_or_clause(self, multiple_companies_metadata):
        """Test that multiple companies are combined with OR."""
        result = build_filter(multiple_companies_metadata)
        assert "(company eq 'Apple' or company eq 'Agilent')" == result

    def test_multiple_years_or_clause(self, multiple_years_metadata):
        """Test that multiple years are combined with OR."""
        result = build_filter(multiple_years_metadata)
        assert result == "(year eq 2024 or year eq 2023)"

    def test_multiple_quarters_or_clause(self, multiple_quarters_metadata):
        """Test that multiple quarters are combined with OR."""
        result = build_filter(multiple_quarters_metadata)
        # Use set comparison since order may vary
        assert "quarter eq 1" in result
        assert "quarter eq 3" in result
        assert "quarter eq 4" in result
        assert result.count(" or ") == 2  # Three items means two OR operators

    def test_multiple_dates_or_clause(self, multiple_dates_metadata):
        """Test that multiple meeting dates are combined with OR."""
        result = build_filter(multiple_dates_metadata)
        assert "(meetingDate eq 2024-01-15T00:00:00Z or meetingDate eq 2024-06-20T00:00:00Z)" == result

    def test_single_earnings_call_complete(self, single_earnings_metadata):
        """Test complete earnings call metadata with single values."""
        result = build_filter(single_earnings_metadata)
        assert "docType eq 'earnings_call'" in result
        assert "company eq 'Apple'" in result
        assert "year eq 2024" in result
        assert "quarter eq 2" in result
        assert " and " in result

    def test_single_meeting_note_complete(self, single_meeting_metadata):
        """Test complete meeting note metadata with single values."""
        result = build_filter(single_meeting_metadata)
        assert "docType eq 'meeting_note'" in result
        assert "author eq 'John'" in result
        assert "meetingDate eq 2024-03-15T00:00:00Z" in result
        assert " and " in result

    def test_complex_metadata_filter(self, complex_metadata):
        """Test complex metadata with multiple fields and values."""
        result = build_filter(complex_metadata)
        
        # Check all fields are present
        assert "docType eq 'earnings_call'" in result
        assert "company eq 'Apple'" in result or "company eq 'Agilent'" in result
        assert "year eq 2024" in result or "year eq 2023" in result
        assert "quarter eq 2" in result or "quarter eq 4" in result
        
        # Check OR clauses for multi-value fields
        assert "(company eq 'Apple' or company eq 'Agilent')" in result
        assert "(year eq 2024 or year eq 2023)" in result
        assert "(quarter eq 2 or quarter eq 4)" in result
        
        # Check fields are joined by AND
        assert result.count(" and ") == 3  # 4 clause groups = 3 AND operators

    def test_mixed_metadata_fields(self, mixed_metadata):
        """Test metadata with both earnings call and meeting note fields."""
        result = build_filter(mixed_metadata)
        
        # Check all fields are present
        assert "docType eq 'earnings_call'" in result
        assert "company eq 'Microsoft'" in result
        assert "year eq 2024" in result
        assert "quarter eq 3" in result
        assert "author eq 'Sarah'" in result
        assert "meetingDate eq 2024-07-15T00:00:00Z" in result

    def test_filter_field_order(self, complex_metadata):
        """Test that filter fields are in expected order."""
        result = build_filter(complex_metadata)
        
        parts = result.split(" and ")
        # Check expected order: docType, company, year, quarter, author, meetingDate
        assert "docType eq 'earnings_call'" in parts[0]
        assert "company" in parts[1]
        assert "year" in parts[2]
        assert "quarter" in parts[3]

    def test_string_fields_have_quotes(self):
        """Test that string fields are wrapped in quotes."""
        metadata = {
            "docType": ["earnings_call"],
            "company": ["Apple"],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": ["John"]
        }
        result = build_filter(metadata)
        
        # String fields should have quotes
        assert "docType eq 'earnings_call'" in result
        assert "company eq 'Apple'" in result
        assert "author eq 'John'" in result

    def test_numeric_fields_no_quotes(self):
        """Test that numeric fields do not have quotes."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [2024],
            "quarter": [2],
            "meetingDate": [],
            "author": []
        }
        result = build_filter(metadata)
        
        # Numeric fields should not have quotes
        assert "year eq 2024" in result
        assert "quarter eq 2" in result

    def test_datetime_fields_no_quotes(self):
        """Test that datetime fields do not have quotes."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-03-15T00:00:00Z"],
            "author": []
        }
        result = build_filter(metadata)
        
        # Datetime fields should not have quotes
        assert "meetingDate eq 2024-03-15T00:00:00Z" in result
        assert "'" not in result  # No quotes at all in this result

    def test_special_characters_in_company_name(self):
        """Test company names with special characters are handled correctly."""
        metadata = {
            "docType": [],
            "company": ["Apple Inc.", "Microsoft & Co."],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": []
        }
        result = build_filter(metadata)
        
        assert "company eq 'Apple Inc.'" in result
        assert "company eq 'Microsoft & Co.'" in result

    def test_special_characters_in_author_name(self):
        """Test author names with special characters are handled correctly."""
        metadata = {
            "docType": [],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": [],
            "author": ["John O'Brien", "Mary-Jane"]
        }
        result = build_filter(metadata)
        
        assert "author eq 'John O'Brien'" in result
        assert "author eq 'Mary-Jane'" in result

    def test_quarter_range(self):
        """Test all possible quarter values."""
        for quarter in range(1, 5):
            metadata = {
                "docType": [],
                "company": [],
                "year": [],
                "quarter": [quarter],
                "meetingDate": [],
                "author": []
            }
            result = build_filter(metadata)
            assert f"quarter eq {quarter}" in result

    def test_realistic_earnings_call_scenario(self):
        """Test realistic earnings call query scenario."""
        metadata = {
            "docType": ["earnings_call"],
            "company": ["Apple", "Agilent"],
            "year": [2024],
            "quarter": [2, 4],
            "meetingDate": [],
            "author": []
        }
        result = build_filter(metadata)
        
        expected_parts = [
            "docType eq 'earnings_call'",
            "(company eq 'Apple' or company eq 'Agilent')",
            "year eq 2024",
            "(quarter eq 2 or quarter eq 4)"
        ]
        
        for part in expected_parts:
            assert part in result

    def test_realistic_meeting_notes_scenario(self):
        """Test realistic meeting notes query scenario."""
        metadata = {
            "docType": ["meeting_note"],
            "company": [],
            "year": [],
            "quarter": [],
            "meetingDate": ["2024-01-15T00:00:00Z", "2024-06-20T00:00:00Z"],
            "author": ["John", "Sarah"]
        }
        result = build_filter(metadata)
        
        assert "docType eq 'meeting_note'" in result
        assert "(meetingDate eq 2024-01-15T00:00:00Z or meetingDate eq 2024-06-20T00:00:00Z)" in result
        assert "(author eq 'John' or author eq 'Sarah')" in result

    def test_multiple_filters_combined_with_and(self, complex_metadata):
        """Test that multiple filter clauses are joined with AND operator."""
        result = build_filter(complex_metadata)
        
        # Count AND operators
        and_count = result.count(" and ")
        
        # Should have 3 AND operators for 4 clause groups
        assert and_count == 3

    def test_filter_consistency(self):
        """Test that build_filter produces consistent results."""
        metadata = {
            "docType": ["earnings_call"],
            "company": ["Apple"],
            "year": [2024],
            "quarter": [2],
            "meetingDate": [],
            "author": []
        }
        
        result1 = build_filter(metadata)
        result2 = build_filter(metadata)
        
        # Results should be identical
        assert result1 == result2

    def test_missing_metadata_keys_handled(self):
        """Test that missing metadata keys don't cause errors."""
        # Metadata with missing keys (should use get with default empty list)
        metadata = {
            "docType": ["earnings_call"],
            "company": ["Apple"]
            # Other keys missing
        }
        
        # Should not raise an error when accessing missing keys
        result = build_filter(metadata)
        assert "docType eq 'earnings_call'" in result
        assert "company eq 'Apple'" in result
