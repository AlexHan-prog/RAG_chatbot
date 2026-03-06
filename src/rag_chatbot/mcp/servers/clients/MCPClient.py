import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from dotenv import load_dotenv
from openai import OpenAI

DEPLOYMENT_NAME = "gpt-5.2-chat"


load_dotenv()

AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            base_url="https://alex-mltg6myf-eastus2.openai.azure.com/openai/v1/",
            api_key=AZURE_OPENAI_API_KEY
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
            "type": "function",
        } for tool in response.tools]

        #print("tools available", available_tools)
        # Initial openAI call this decides which tool(s) should be used
        response = self.client.responses.create(
            model=DEPLOYMENT_NAME,
            input=messages,
            tools=available_tools,
            max_output_tokens=1000
        )

        # Process response and handle tool calls
        final_text = []

        # This only looks at one tool not multiple
        assistant_message_content = []
        #print("response.output:", response.output)
        for item in response.output:
            print(item)
            if item.type == 'message':
                final_text.append(item.text)
                assistant_message_content.append(content)
            elif content.type == 'function_call':
                tool_name = content.name
                tool_args = json.loads(content.arguments)


                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Send response from calling tools to the model (call model again)
                # For example the model will receive here that from using the tools 
                # a new task was created 
                response = self.client.responses.create(
                    model=DEPLOYMENT_NAME,
                    input=messages,
                    tools=available_tools,
                    max_output_tokens=1000
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    path_to_server = r"C:\Users\alexh\Desktop\LLM_uni_project\RAG_chatbot\src\rag_chatbot\mcp\servers\jira_server.py"
    #path_to_server = "src\\rag_chatbot\\mcp\\servers\\jira_server.py"
    # if len(sys.argv) < 2:
    #     print("Usage: python client.py")
    #     sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(path_to_server)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())