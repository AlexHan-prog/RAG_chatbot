import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.backend.mcp.servers.clients.MCPClient import MCPClient, DEPLOYMENT_NAME


# ==================== FIXTURES ====================


@pytest.fixture
def event_loop():
	"""Use a fresh event loop for these async tests (pytest override)."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
def mcp_client():
	"""Return a fresh MCPClient instance for each test."""
	return MCPClient()


# ==================== TEST __init__ ====================


class TestInit:
	def test_initial_state(self, mcp_client):
		"""__init__ should set session to None and create an AsyncExitStack."""
		assert mcp_client.session is None
		assert mcp_client.exit_stack is not None
		# Client attribute should be set (imported from env)
		assert mcp_client.client is not None


# ==================== TEST connect_to_server ====================


class TestConnectToServer:
	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.clients.MCPClient.ClientSession")
	@patch("src.rag_chatbot.mcp.servers.clients.MCPClient.stdio_client")
	async def test_connect_to_server_initializes_session_and_lists_tools(
		self,
		mock_stdio_client,
		mock_client_session,
		mcp_client,
	):
		"""connect_to_server should start stdio client, create session, initialize, and list tools."""
		# Mock stdio_client context manager
		stdio_mock = AsyncMock()
		stdio_transport = (MagicMock(), MagicMock())
		stdio_mock.__aenter__.return_value = stdio_transport
		mock_stdio_client.return_value = stdio_mock

		# Mock ClientSession context manager
		session_instance = AsyncMock()
		tools_response = MagicMock()
		tools_response.tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
		session_instance.list_tools.return_value = tools_response
		session_instance.initialize = AsyncMock()

		session_cm = AsyncMock()
		session_cm.__aenter__.return_value = session_instance
		mock_client_session.return_value = session_cm

		await mcp_client.connect_to_server("rag_chatbot.mcp.servers.jira_server")

		# stdio_client should be called with StdioServerParameters
		mock_stdio_client.assert_called_once()

		# ClientSession should be created with stdio + write
		mock_client_session.assert_called_once()

		# Session.initialize and list_tools should be called
		session_instance.initialize.assert_awaited_once()
		session_instance.list_tools.assert_awaited_once()

		# The client's session should hold the created session instance
		assert mcp_client.session is session_instance


# ==================== TEST process_query ====================


class TestProcessQuery:
	@pytest.mark.asyncio
	async def test_process_query_no_tool_calls_returns_aggregated_text(self, mcp_client):
		"""When the model returns only message output, process_query should aggregate text and stop."""
		# Prepare a fake session on the client
		session = AsyncMock()
		session.list_tools.return_value = MagicMock(tools=[])
		mcp_client.session = session

		# Fake LLM response with a single message and no tool calls
		message_part = MagicMock()
		message_part.type = "output_text"
		message_part.text = "Hello from the model."

		message_item = MagicMock()
		message_item.type = "message"
		message_item.content = [message_part]

		response = MagicMock()
		response.output = [message_item]
		response.id = "resp-1"

		client_mock = MagicMock()
		client_mock.responses.create.return_value = response
		mcp_client.client = client_mock

		result = await mcp_client.process_query("hi")

		# list_tools should be awaited
		session.list_tools.assert_awaited_once()

		# LLM should be called once because there are no tool calls
		client_mock.responses.create.assert_called_once()
		args, kwargs = client_mock.responses.create.call_args
		assert kwargs["model"] == DEPLOYMENT_NAME
		assert kwargs["input"] == "hi"

		assert result == "Hello from the model."

	@pytest.mark.asyncio
	async def test_process_query_with_tool_call_then_followup_message(self, mcp_client):
		"""If the model issues a tool call, MCPClient should call the MCP tool, then continue until a final message is produced."""
		# Setup mocked tools from session
		session = AsyncMock()
		tool = MagicMock()
		tool.name = "example_tool"
		tool.description = "desc"
		tool.inputSchema = {"type": "object", "properties": {}}
		session.list_tools.return_value = MagicMock(tools=[tool])

		# When tool is called, return an object with 'content'
		tool_result = MagicMock()
		tool_result.content = {"result": "tool-output"}
		session.call_tool.return_value = tool_result
		mcp_client.session = session

		# First LLM response issues a tool call
		tool_call_item = MagicMock()
		tool_call_item.type = "function_call"
		tool_call_item.name = "example_tool"
		tool_call_item.arguments = json.dumps({"foo": "bar"})
		tool_call_item.call_id = "call-1"

		first_response = MagicMock()
		first_response.output = [tool_call_item]
		first_response.id = "resp-1"

		# Second LLM response returns a message using the tool output
		message_part = MagicMock()
		message_part.type = "output_text"
		message_part.text = "Final answer after tool call."

		message_item = MagicMock()
		message_item.type = "message"
		message_item.content = [message_part]

		second_response = MagicMock()
		second_response.output = [message_item]
		second_response.id = "resp-2"

		client_mock = MagicMock()
		# First call -> tool call, second call -> final message
		client_mock.responses.create.side_effect = [first_response, second_response]
		mcp_client.client = client_mock

		result = await mcp_client.process_query("run with tools")

		# Session tools listing
		session.list_tools.assert_awaited_once()

		# Tool should be called once with parsed arguments
		session.call_tool.assert_awaited_once_with("example_tool", {"foo": "bar"})

		# LLM should be invoked twice: initial query + follow-up with tool output
		assert client_mock.responses.create.call_count == 2

		first_call_kwargs = client_mock.responses.create.call_args_list[0].kwargs
		assert first_call_kwargs["input"] == "run with tools"

		second_call_kwargs = client_mock.responses.create.call_args_list[1].kwargs
		assert second_call_kwargs["previous_response_id"] == "resp-1"
		# Follow-up input should contain the function_call_output structure
		followup_input = second_call_kwargs["input"]
		assert isinstance(followup_input, list)
		assert followup_input[0]["type"] == "function_call_output"

		assert result == "Final answer after tool call."


# ==================== TEST cleanup ====================


class TestCleanup:
	@pytest.mark.asyncio
	async def test_cleanup_closes_exit_stack(self, mcp_client):
		"""cleanup should aclose the AsyncExitStack."""
		mcp_client.exit_stack = AsyncMock()

		await mcp_client.cleanup()

		mcp_client.exit_stack.aclose.assert_awaited_once()

