import pytest
from unittest.mock import patch, Mock, MagicMock
from types import SimpleNamespace

from src.backend.rag.blob_utils import (
	contextual_chunking,
	chunk_epics,
	chunk_from_blob,
)


# ==================== FIXTURES ====================


@pytest.fixture
def sample_text():
	"""Sample full document text for chunking tests."""
	return "This is the full transcript text used as context."


@pytest.fixture
def sample_chunks():
	"""Sample list of raw chunks."""
	return ["chunk one", "chunk two"]


@pytest.fixture
def mock_blob_text():
	"""Sample blob text content for blob chunking tests."""
	return "Line 1 of the document.\nLine 2 of the document."


@pytest.fixture
def mock_txt_blob():
	"""A simple mock blob object representing a .txt file."""
	return SimpleNamespace(name="file1.txt")


@pytest.fixture
def mock_non_txt_blob():
	"""A simple mock blob object representing a non-text file."""
	return SimpleNamespace(name="image.png")


@pytest.fixture
def mock_container_client(mock_txt_blob, mock_non_txt_blob, mock_blob_text):
	"""Mock Azure Blob container client with one text blob and one non-text blob."""
	container = MagicMock()

	# list_blobs returns both text and non-text blobs; function should skip non-text
	container.list_blobs.return_value = [mock_non_txt_blob, mock_txt_blob]

	blob_client = MagicMock()
	download_stream = MagicMock()
	download_stream.readall.return_value = mock_blob_text.encode("utf-8")
	blob_client.download_blob.return_value = download_stream

	# get_blob_client is called with the blob object; return same client for simplicity
	container.get_blob_client.return_value = blob_client

	return container


# ==================== TEST CONTEXTUAL_CHUNKING ====================


class TestContextualChunking:
	"""Unit tests for contextual_chunking function."""

	@patch("src.rag_chatbot.rag.blob_utils.LLMChunker")
	def test_prepends_llm_context_to_each_chunk(self, mock_llm_chunker_cls, sample_text, sample_chunks):
		"""Each chunk should be prefixed with LLM-generated context."""
		mock_llm_chunker = Mock()

		def side_effect(document, chunk):
			return f"context-for-{chunk}"

		mock_llm_chunker.return_response.side_effect = side_effect
		mock_llm_chunker_cls.return_value = mock_llm_chunker

		result = contextual_chunking(sample_text, sample_chunks)

		assert len(result) == len(sample_chunks)
		assert result[0] == "context-for-chunk one chunk one"
		assert result[1] == "context-for-chunk two chunk two"

	@patch("src.rag_chatbot.rag.blob_utils.LLMChunker")
	def test_calls_llm_chunker_with_full_document_and_each_chunk(self, mock_llm_chunker_cls, sample_text, sample_chunks):
		"""LLMChunker.return_response should receive the full text and each chunk once."""
		mock_llm_chunker = Mock()
		mock_llm_chunker.return_response.return_value = "ctx"
		mock_llm_chunker_cls.return_value = mock_llm_chunker

		contextual_chunking(sample_text, sample_chunks)

		# One call per chunk
		assert mock_llm_chunker.return_response.call_count == len(sample_chunks)
		for call in mock_llm_chunker.return_response.call_args_list:
			assert call.kwargs["document"] == sample_text
			assert "chunk" in call.kwargs


# ==================== TEST CHUNK_EPICS ====================


class TestChunkEpics:
	"""Unit tests for chunk_epics function."""

	def test_returns_empty_list_when_no_epic_headings(self):
		"""If there are no 'Epic' headings, return an empty list."""
		notes = """Meeting notes without any special heading.\nJust some text here."""

		result = chunk_epics(notes)

		assert result == []

	def test_splits_meeting_notes_into_epic_chunks(self):
		"""Multiple Epic sections should be split into separate chunks."""
		notes = """Epic 1: First epic\nLine A\nLine B\n\nEpic 2 - Second epic\nLine C\nLine D\n"""

		chunks = chunk_epics(notes)

		assert len(chunks) == 2
		# First chunk
		assert chunks[0].startswith("Epic 1: First epic")
		assert "Line A" in chunks[0]
		assert "Line B" in chunks[0]
		assert "Epic 2 - Second epic" not in chunks[0]
		# Second chunk
		assert chunks[1].startswith("Epic 2 - Second epic")
		assert "Line C" in chunks[1]
		assert "Line D" in chunks[1]

	def test_matches_case_insensitive_epic_headings(self):
		"""Headings like 'epic 1:' should also be detected (case-insensitive)."""
		notes = """epic 1: lower case heading\nSome content here."""

		chunks = chunk_epics(notes)

		assert len(chunks) == 1
		assert chunks[0].startswith("epic 1: lower case heading")


# ==================== TEST CHUNK_FROM_BLOB ====================


class TestChunkFromBlob:
	"""Unit tests for chunk_from_blob function."""

	@patch("src.rag_chatbot.rag.blob_utils.RecursiveCharacterTextSplitter")
	def test_basic_chunking_skips_non_text_blobs(
		self,
		mock_splitter_cls,
		mock_container_client,
		mock_blob_text,
	):
		"""Non-.txt blobs are skipped; text blobs are split into chunks."""
		splitter_instance = Mock()
		splitter_instance.split_text.return_value = ["chunk-1", "chunk-2"]
		mock_splitter_cls.return_value = splitter_instance

		result = chunk_from_blob(
			container_client=mock_container_client,
			doc_type="earnings_call",
			chunk_size=100,
			context_chunking=False,
			overlap=False,
			epic_chunking=False,
		)

		mock_container_client.list_blobs.assert_called_once()
		mock_splitter_cls.assert_called_once_with(chunk_size=100, chunk_overlap=0)
		splitter_instance.split_text.assert_called_once_with(mock_blob_text)

		assert len(result) == 2
		assert {"source", "chunk_id", "content", "docType"} <= result[0].keys()
		assert result[0]["source"] == "file1.txt"
		assert result[0]["chunk_id"] == 0
		assert result[0]["docType"] == "earnings_call"

	@patch("src.rag_chatbot.rag.blob_utils.RecursiveCharacterTextSplitter")
	def test_overlap_sets_chunk_overlap_to_ten_percent(
		self,
		mock_splitter_cls,
		mock_container_client,
	):
		"""When overlap=True, chunk_overlap should be 10% of chunk_size (integer)."""
		splitter_instance = Mock()
		splitter_instance.split_text.return_value = ["only-chunk"]
		mock_splitter_cls.return_value = splitter_instance

		chunk_size = 120
		chunk_from_blob(
			container_client=mock_container_client,
			doc_type="earnings_call",
			chunk_size=chunk_size,
			context_chunking=False,
			overlap=True,
			epic_chunking=False,
		)

		mock_splitter_cls.assert_called_once_with(
			chunk_size=chunk_size,
			chunk_overlap=int(chunk_size * 0.1),
		)

	@patch("src.rag_chatbot.rag.blob_utils.contextual_chunking")
	@patch("src.rag_chatbot.rag.blob_utils.RecursiveCharacterTextSplitter")
	def test_context_chunking_applies_llm_context(
		self,
		mock_splitter_cls,
		mock_contextual_chunking,
		mock_container_client,
		mock_blob_text,
	):
		"""When context_chunking=True, contextual_chunking should be called and its output used."""
		splitter_instance = Mock()
		splitter_instance.split_text.return_value = ["base-chunk"]
		mock_splitter_cls.return_value = splitter_instance

		mock_contextual_chunking.return_value = ["ctx-base-chunk"]

		result = chunk_from_blob(
			container_client=mock_container_client,
			doc_type="earnings_call",
			chunk_size=100,
			context_chunking=True,
			overlap=False,
			epic_chunking=False,
		)

		mock_contextual_chunking.assert_called_once_with(
			text=mock_blob_text,
			chunks=["base-chunk"],
		)

		assert len(result) == 1
		assert result[0]["content"] == "ctx-base-chunk"

	@patch("src.rag_chatbot.rag.blob_utils.chunk_epics")
	@patch("src.rag_chatbot.rag.blob_utils.contextual_chunking")
	@patch("src.rag_chatbot.rag.blob_utils.RecursiveCharacterTextSplitter")
	def test_meeting_notes_with_epic_and_context_chunking(
		self,
		mock_splitter_cls,
		mock_contextual_chunking,
		mock_chunk_epics,
		mock_container_client,
		mock_blob_text,
	):
		"""For doc_type='meeting_note', epic and context chunking should both be applied."""
		splitter_instance = Mock()
		splitter_instance.split_text.return_value = ["base-chunk"]
		mock_splitter_cls.return_value = splitter_instance

		mock_contextual_chunking.return_value = ["ctx-base-chunk"]
		mock_chunk_epics.return_value = ["epic-1", "epic-2"]

		result = chunk_from_blob(
			container_client=mock_container_client,
			doc_type="meeting_note",
			chunk_size=100,
			context_chunking=True,
			overlap=False,
			epic_chunking=True,
		)

		mock_contextual_chunking.assert_called_once_with(
			text=mock_blob_text,
			chunks=["base-chunk"],
		)
		mock_chunk_epics.assert_called_once_with(mock_blob_text)

		contents = [r["content"] for r in result]
		assert contents == ["ctx-base-chunk", "epic-1", "epic-2"]
		assert all(r["docType"] == "meeting_note" for r in result)

	@patch("src.rag_chatbot.rag.blob_utils.chunk_epics")
	@patch("src.rag_chatbot.rag.blob_utils.RecursiveCharacterTextSplitter")
	def test_epic_chunking_disabled_for_non_meeting_notes(
		self,
		mock_splitter_cls,
		mock_chunk_epics,
		mock_container_client,
	):
		"""Even if epic_chunking=True, non-meeting_note doc types should not call chunk_epics."""
		splitter_instance = Mock()
		splitter_instance.split_text.return_value = ["base-chunk"]
		mock_splitter_cls.return_value = splitter_instance

		result = chunk_from_blob(
			container_client=mock_container_client,
			doc_type="earnings_call",  # not 'meeting_note'
			chunk_size=100,
			context_chunking=False,
			overlap=False,
			epic_chunking=True,
		)

		mock_chunk_epics.assert_not_called()
		assert [r["content"] for r in result] == ["base-chunk"]

	@patch("src.rag_chatbot.rag.blob_utils.RecursiveCharacterTextSplitter")
	def test_returns_empty_list_when_no_text_blobs(
		self,
		mock_splitter_cls,
	):
		"""If container has no .txt blobs, result should be empty list."""
		container = MagicMock()
		container.list_blobs.return_value = [SimpleNamespace(name="image.png")]

		result = chunk_from_blob(
			container_client=container,
			doc_type="earnings_call",
			chunk_size=100,
			context_chunking=False,
			overlap=False,
			epic_chunking=False,
		)

		# Splitter should still be created once but never used
		mock_splitter_cls.assert_called_once()
		assert result == []

