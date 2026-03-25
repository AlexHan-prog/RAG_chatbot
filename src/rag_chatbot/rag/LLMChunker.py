from src.rag_chatbot.rag.env import client, deployment_name


class LLMChunker:
    """
    Uses an LLM to generate contextual summaries for document chunks.

    The purpose of this class is to enrich each chunk with additional
    retrieval-oriented context, improving search relevance in downstream
    RAG (Retrieval-Augmented Generation) pipelines.

    The generated context helps:
    - Identify what the chunk is about
    - Place it within the structure of the full document
    - Improve semantic search and ranking
    """


    instructions = """
    <document>
    {WHOLE_DOCUMENT}
    </document>

    <chunk>
    {CHUNK_CONTENT}
    </chunk>

    Write 1–2 sentences of retrieval-oriented context for this chunk.

    The context should identify:
    1. the main topic of the chunk,
    2. where it sits in the document hierarchy,
    3. any parent epic or story if explicitly present in the document,
    4. the type of content (e.g. story description, acceptance criteria, implementation note, meeting discussion, action item).

    Prefer explicit document structure over vague summary.
    Do not invent missing hierarchy.
    Return only the context. 
    """
    
    def __init__(self):
        """initialises chunker with pre-configured client"""
        self.client = client

    def return_response(self, document: str, chunk: str) -> str:
        """
        returns a brief description about the chunk passed in and where it belongs in relative to the entire document

        Args:
            document (str): Entire text document
            chunk (str): chunk that belongs in the document
        Returns:
            str: context about chunk in relation to the entire document it is part of
        """
        msg = self.instructions.format(
            WHOLE_DOCUMENT=document,
            CHUNK_CONTENT=chunk)

        response = self.client.responses.parse(
            model=deployment_name,
            input=msg,
        )
        return response.output_text