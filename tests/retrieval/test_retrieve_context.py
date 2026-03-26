import pytest
from unittest.mock import patch, Mock
from src.backend.rag.retrieval_utils import retrieve_context


class TestRetrieveContext:
    """Unit tests for retrieve_context function."""

    # ============== FIXTURES ==============

    @pytest.fixture
    def mock_vector_embedding(self):
        """Fixture for mock vector embedding."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.fixture
    def mock_filter_metadata(self):
        """Fixture for mock filter metadata."""
        return {
            "docType": ["earnings_call"],
            "company": ["Apple"],
            "year": [2024],
            "quarter": [2],
            "meetingDate": [],
            "author": []
        }

    @pytest.fixture
    def mock_route_transcripts(self):
        """Fixture for transcripts routing."""
        return Mock(source="transcripts")

    @pytest.fixture
    def mock_route_meeting_notes(self):
        """Fixture for meeting notes routing."""
        return Mock(source="meeting_notes")

    @pytest.fixture
    def mock_route_both(self):
        """Fixture for both routing."""
        return Mock(source="both")

    # ============== TESTS ==============

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_returns_list(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context returns a list."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        result = retrieve_context("test query")

        assert isinstance(result, list)

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_generates_embeddings(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context generates embeddings for the query."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        query = "test query"
        retrieve_context(query)

        mock_generate_embeddings.assert_called_once_with([query])

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_calls_route_query(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context calls route_query."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        query = "test query"
        retrieve_context(query)

        mock_route_query.assert_called_once_with(query)

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_retrieves_filter_metadata(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context calls retrieve_filter_metadata."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        query = "test query"
        retrieve_context(query)

        mock_retrieve_filter_metadata.assert_called_once_with(query)

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_searches_transcripts_only(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_meeting_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context searches only transcripts when routed."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        retrieve_context("test query")

        mock_transcript_client.search.assert_called_once()
        mock_meeting_client.search.assert_not_called()

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_searches_meeting_notes_only(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_meeting_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_meeting_notes,
    ):
        """Test that retrieve_context searches only meeting_notes when routed."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_meeting_notes
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_meeting_client.search.return_value = []

        retrieve_context("test query")

        mock_meeting_client.search.assert_called_once()
        mock_transcript_client.search.assert_not_called()

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_searches_both_indexes(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_meeting_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_both,
    ):
        """Test that retrieve_context searches both indexes when routed."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_both
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []
        mock_meeting_client.search.return_value = []

        retrieve_context("test query")

        mock_transcript_client.search.assert_called_once()
        mock_meeting_client.search.assert_called_once()

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_respects_k_parameter(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context passes k parameter to search."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        k = 10
        retrieve_context("test query", k=k)

        call_args = mock_transcript_client.search.call_args
        assert call_args.kwargs["top"] == k

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_uses_default_k(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context uses default k=6 when not specified."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        retrieve_context("test query")

        call_args = mock_transcript_client.search.call_args
        assert call_args.kwargs["top"] == 6

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_empty_results(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context handles empty search results."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        result = retrieve_context("test query")

        assert result == []

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.create_safe_filter_for_index")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_calls_create_safe_filter_for_index(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_create_safe_filter,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context calls create_safe_filter_for_index."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_create_safe_filter.return_value = ""
        mock_transcript_client.search.return_value = []

        retrieve_context("test query")

        # Should be called for transcripts
        assert mock_create_safe_filter.call_count >= 1
        call_args = mock_create_safe_filter.call_args_list[0]
        assert call_args[0][1] == "transcripts"

    @patch("src.rag_chatbot.rag.retrieval_utils.MEETING_NOTES_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.TRANSCRIPT_SEARCH_CLIENT")
    @patch("src.rag_chatbot.rag.retrieval_utils.retrieve_filter_metadata")
    @patch("src.rag_chatbot.rag.retrieval_utils.route_query")
    @patch("src.rag_chatbot.rag.retrieval_utils.generate_embeddings")
    def test_retrieve_context_passes_query_text_to_search(
        self,
        mock_generate_embeddings,
        mock_route_query,
        mock_retrieve_filter_metadata,
        mock_transcript_client,
        mock_vector_embedding,
        mock_filter_metadata,
        mock_route_transcripts,
    ):
        """Test that retrieve_context passes query text to search."""
        mock_generate_embeddings.return_value = [mock_vector_embedding]
        mock_route_query.return_value = mock_route_transcripts
        mock_retrieve_filter_metadata.return_value = mock_filter_metadata
        mock_transcript_client.search.return_value = []

        query = "Apple earnings Q2 2024"
        retrieve_context(query)

        call_args = mock_transcript_client.search.call_args
        assert call_args.kwargs["search_text"] == query
