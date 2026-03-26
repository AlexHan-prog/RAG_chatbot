import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.backend.rag.RAG_bot import MCPLLM, HISTORY_LEN, deployment_name

@pytest.fixture
def mock_mcp_client():
    with patch("src.rag_chatbot.rag.RAG_bot.MCPClient") as mock:
        yield mock


@pytest.mark.asyncio
async def test_generate_answer_builds_messages(mock_mcp_client):
    mock_client = mock_mcp_client.return_value # mock_mcp_client.return_value == MCPClient() as 
    mock_client.connect_to_server = AsyncMock()
    mock_client.process_query = AsyncMock(return_value="response")

    llm = MCPLLM()

    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]

    result = await llm.generate_answer("How are you?", history)

    # Check response
    assert result == "response"

    # Check messages passed to MCP
    called_args = mock_client.process_query.call_args.kwargs["query"]

    assert called_args[-1] == {"role": "user", "content": "How are you?"}

def test_mcpllm_initialization():
    llm = MCPLLM()

    assert llm is not None
    assert llm.connected is False
    assert llm.client is not None

@pytest.mark.asyncio
async def test_connect_to_server_sets_connected_true():
    mock_client = MagicMock()
    mock_client.connect_to_server = AsyncMock()

    llm = MCPLLM()
    llm.client = mock_client

    await llm.connect_to_MCPserver()

    mock_client.connect_to_server.assert_called_once_with(llm.server_module)
    assert llm.connected is True

@pytest.mark.asyncio
async def test_connect_does_not_reconnect_if_already_connected():
    mock_client = MagicMock()
    mock_client.connect_to_server = AsyncMock()

    llm = MCPLLM()
    llm.client = mock_client
    llm.connected = True

    await llm.connect_to_MCPserver()

    mock_client.connect_to_server.assert_not_called()

@pytest.mark.asyncio
async def test_connect_raises_on_failure():
    mock_client = MagicMock()
    mock_client.connect_to_server = AsyncMock(side_effect=Exception("fail"))

    llm = MCPLLM()
    llm.client = mock_client

    with pytest.raises(Exception):
        await llm.connect_to_MCPserver()

    assert llm.connected is False

@pytest.mark.asyncio
async def test_generate_answer_calls_connect():
    llm = MCPLLM()

    llm.connect_to_MCPserver = AsyncMock()
    llm.client = MagicMock()
    llm.client.process_query = AsyncMock(return_value="ok")

    await llm.generate_answer("Hello", [])

    llm.connect_to_MCPserver.assert_called_once()

@pytest.mark.asyncio
async def test_generate_answer_respects_history_limit():
    """
    Tests that the message list sent to process_query is 1 + HISTORY_LEN,
    the extra 1 coming from the user_query that was just entered.
    """
    mock_client = MagicMock()
    mock_client.connect_to_server = AsyncMock()
    mock_client.process_query = AsyncMock(return_value="response")

    llm = MCPLLM()
    llm.client = mock_client

    history = [
        {"role": "user", "content": f"msg{i}"}
        for i in range(10)
    ]

    
    await llm.generate_answer("new", history)

    called_messages = mock_client.process_query.call_args[1]["query"]

    assert len(called_messages) == min(len(history), HISTORY_LEN) + 1

@pytest.mark.asyncio
async def test_cleanup_calls_client_and_resets_flag():
    mock_client = MagicMock()
    mock_client.cleanup = AsyncMock()

    llm = MCPLLM()
    llm.client = mock_client
    llm.connected = True

    await llm.cleanup()

    mock_client.cleanup.assert_called_once()
    assert llm.connected is False

@pytest.mark.asyncio
async def test_cleanup_does_nothing_if_not_connected():
    mock_client = MagicMock()
    mock_client.cleanup = AsyncMock()

    llm = MCPLLM()
    llm.client = mock_client
    llm.connected = False

    await llm.cleanup()

    mock_client.cleanup.assert_not_called()

@pytest.mark.asyncio
async def test_generate_answer_fails_if_connection_fails():
    """
    If the connection to the MCP server fails when trying to generate an answer,
    process_query should not be called.
    """
    llm = MCPLLM()

    # Mock connect_to_MCPserver to fail
    llm.connect_to_MCPserver = AsyncMock(side_effect=Exception("connection failed"))

    # Mock client
    llm.client = MagicMock()
    llm.client.process_query = AsyncMock()

    with pytest.raises(Exception, match="connection failed"):
        await llm.generate_answer("Hello", [])

    # Ensure process_query was never called
    llm.client.process_query.assert_not_called()