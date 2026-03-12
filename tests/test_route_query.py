import pytest
from src.rag_chatbot.rag.retrieval_utils import route_query


@pytest.mark.parametrize(
    "query,expected_source",
    [
        (
            """
            From Reuben's meeting notes on the 28th January 2026,
            the team discussed upcoming product milestones, internal tooling improvements,
            and priorities for the next development sprint.
            """,
            "meeting_notes",
        ),
        (
            """
            During the Agilent Technologies Q2 2024 earnings call,
            which executives were present for the prepared remarks
            and who joined for the Q&A portion?
            """,
            "transcripts",
        ),
        (
            """
            Summarise discussions about product strategy and roadmap across both
            internal meeting notes and company transcripts.
            """,
            "both",
        ),
    ],
)
def test_route_query(query, expected_source):
    route = route_query(query)
    assert route.source == expected_source