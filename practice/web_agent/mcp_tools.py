"""MCP (Model Context Protocol) tools integration for web crawling."""

import os
import asyncio
from typing import List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool


class MCPToolManager:
    """Manages MCP tools for web crawling operations."""

    def __init__(
        self,
        mcp_server: str = "firecrawl-mcp",
        firecrawl_api_key: Optional[str] = None,
    ):
        """
        Initialize MCP Tool Manager.

        Args:
            mcp_server: Name of the MCP server to use (default: "firecrawl-mcp")
            firecrawl_api_key: Firecrawl API key (if None, reads from env)
        """
        self.mcp_server = mcp_server
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.session: Optional[ClientSession] = None
        self.tools: List[BaseTool] = []

    def _get_server_params(self) -> StdioServerParameters:
        """Get server parameters for MCP server."""
        env = {}
        if self.firecrawl_api_key:
            env["FIRECRAWL_API_KEY"] = self.firecrawl_api_key

        return StdioServerParameters(
            command="npx",
            env=env,
            args=[self.mcp_server],
        )

    async def initialize(self) -> List[BaseTool]:
        """
        Initialize MCP connection and load tools.

        Returns:
            List of LangChain tools from MCP server
        """
        server_params = self._get_server_params()

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                # Store tools for later use
                self.tools = tools
                return tools

    async def get_tools(self) -> List[BaseTool]:
        """
        Get available MCP tools.

        Returns:
            List of available tools
        """
        if not self.tools:
            self.tools = await self.initialize()
        return self.tools

    def get_tool_names(self) -> List[str]:
        """Get names of available tools."""
        return [tool.name for tool in self.tools] if self.tools else []


async def create_mcp_tools(
    mcp_server: str = "firecrawl-mcp",
    firecrawl_api_key: Optional[str] = None,
) -> List[BaseTool]:
    """
    Convenience function to create MCP tools.

    Args:
        mcp_server: Name of the MCP server
        firecrawl_api_key: Firecrawl API key

    Returns:
        List of LangChain tools
    """
    manager = MCPToolManager(mcp_server, firecrawl_api_key)
    return await manager.get_tools()
