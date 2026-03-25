from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag_chatbot.rag.LLMChunker import LLMChunker
import re



def contextual_chunking(text: str, chunks: list[str]) -> list[str]:
    """
    Enhance chunks with additional LLM-generated context.

    For each chunk, this function:
    1. Uses the full document as global context
    2. Generates a short description of the chunk via LLM
    3. Prepends the generated context to the chunk content

    This improves retrieval quality by making each chunk more
    semantically meaningful when embedded.

    Args:
        text (str):
            The full document text (e.g. full transcript or meeting notes)

        chunks (list[str]):
            List of raw text chunks (without context)

    Returns:
        list[str]:
            List of chunks where each chunk is prefixed with
            its generated contextual description
    """

    llm_chunker = LLMChunker()
    contextual_chunks = []
    
    for c in chunks:
        context = llm_chunker.return_response(document=text, chunk=c)
        contextual_chunks.append(context + " " + c)
    
    return contextual_chunks

def chunk_epics(meeting_notes: str) -> list[str]:
    """
    Split meeting notes into chunks based on "Epic" sections.

    Each chunk:
    - Starts at a line matching "Epic <number>: <title>"
    - Ends at the next Epic heading or end of document

    Example headings supported:
        Epic 1: Title
        Epic 2 - Dashboard

    Args:
        meeting_notes (str):
            Full meeting notes text

    Returns:
        list[str]:
            List of epic-based chunks, where each chunk contains:
            - the epic heading
            - the full text under that epic
    """

    # Matches lines like:
    # Epic 1:
    # Epic 2: Dashboard
    # Epic 10 - Some title   (if you want to support '-' too)
    epic_pattern = re.compile(
        r'^(Epic\s+\d+\s*[:\-].*)$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(epic_pattern.finditer(meeting_notes))

    if not matches:
        return []

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(meeting_notes)

        chunk_text = meeting_notes[start:end].strip()
        
        # chunk_text includes the name of the epic 
        chunks.append(chunk_text)

    return chunks

def chunk_from_blob(
        container_client,
        doc_type: str,
        chunk_size: int=756,
        context_chunking: bool=False,
        overlap: bool=False,
        epic_chunking: bool=False) -> list[dict]: 
    """
    Extract and chunk text documents from an Azure Blob container.

    This function:
    1. Iterates over all blobs in the container
    2. Downloads and decodes text files
    3. Splits text into chunks using a recursive splitter
    4. Optionally augments chunks with:
        - LLM-generated context (context_chunking)
        - Epic-based structure (epic_chunking)
    5. Returns structured chunk data for indexing

    Args:
        container_client:
            Azure Blob container client used to list and retrieve blobs

        doc_type (str):
            Type of document (e.g. "earnings_call", "meeting_note")

        chunk_size (int, optional):
            Maximum size of each chunk (in characters)

        context_chunking (bool, optional):
            If True, prepend LLM-generated context to each chunk

        overlap (bool, optional):
            If True, apply ~10% overlap between chunks to preserve context

        epic_chunking (bool, optional):
            If True, additionally extract structured "Epic"-based chunks

    Returns:
        list[dict]:
            List of chunk dictionaries, each containing:
            {
                "source": str,     # blob filename
                "chunk_id": int,  # index of chunk within document
                "content": str,   # chunk text (possibly contextualised)
                "docType": str    # document type
            }
    
    """

    # epic_chunking should not occur for documents other than meeting notes
    if doc_type != "meeting_note":
        epic_chunking = False

    overlap = int(chunk_size * 0.1) if overlap else 0 # overlap should be 10-20% of chunks size

    # retrieve and sort blobs by name for deterministic processing
    # A container is a folder and a blob is a file, so here we are listing all the .txt files in a specific folder (container)
    blobs = sorted(container_client.list_blobs(), key=lambda b:b.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap)

    transcript_chunks = []

    for blob in blobs:
        # If the file is not a text file we can't handle it
        if not blob.name.lower().endswith(".txt"):
            continue

        blob_client = container_client.get_blob_client(blob)

        download_stream = blob_client.download_blob()
       
        transcript_text = download_stream.readall().decode("utf-8") # get text in file 
     
        chunks = text_splitter.split_text(transcript_text) # Applies recursive text splitting to text

        if context_chunking:
            # replace chunks with contextual chunks (append context)
            chunks = contextual_chunking(text=transcript_text, chunks=chunks)

        if epic_chunking:
            epic_chunks = chunk_epics(transcript_text)
            # epic_chunks are chunks only containing the epics, so just extend them onto the normal chunks obtained
            chunks.extend(epic_chunks)

        for j, chunk in enumerate(chunks):
            transcript_chunks.append({
                "source": blob.name,
                "chunk_id": j,
                "content": chunk,
                "docType": doc_type
            })

    return transcript_chunks
