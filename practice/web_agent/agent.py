"""Main web agent implementation using LangGraph."""

import os
import asyncio
from typing import Optional, Dict, Any, List
from unittest import skip
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from mcp_tools import MCPToolManager
from summarizer import Summarizer

load_dotenv()


class LocalOllama:
    llm = ChatOpenAI(
        model="llama3.2:3b",
        temperature=0,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    def __init__(self):
        skip


class WebAgent:
    """Web agent that queries websites and summarizes information."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        openai_api_key: Optional[str] = None,
        firecrawl_api_key: Optional[str] = None,
        mcp_server: str = "firecrawl-mcp",
    ):
        """
        Initialize Web Agent.

        Args:
            model_name: Name of the LLM model
            temperature: Temperature for model generation
            openai_api_key: OpenAI API key
            firecrawl_api_key: Firecrawl API key
            mcp_server: MCP server name
        """
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.mcp_server = mcp_server

        # Initialize components
        self.llm = LocalOllama.llm

        self.mcp_manager = MCPToolManager(
            mcp_server=mcp_server,
            firecrawl_api_key=self.firecrawl_api_key,
        )
        self.summarizer = Summarizer(
            model_name=model_name,
            temperature=temperature,
            api_key=self.openai_api_key,
        )

        # Agent will be initialized when tools are loaded
        self.agent = None
        self.tools: List = []

    async def _initialize_agent(self):
        """Initialize the agent with MCP tools."""
        if self.agent is None:
            # Load MCP tools
            self.tools = await self.mcp_manager.get_tools()
            # Create agent with tools
            self.agent = create_agent(self.llm, self.tools)

    async def query_website(
        self,
        url: str,
        query: str,
        summarize: bool = True,
    ) -> Dict[str, Any]:
        """
        Query a website and optionally summarize the results.

        Args:
            url: URL of the website to query
            query: Query/question about the website
            summarize: Whether to summarize the results

        Returns:
            Dictionary with query results and optional summary
        """
        await self._initialize_agent()

        # Create system message
        system_message = SystemMessage(
            content="""
            You are a helpful web assistant that can scrape websites, crawl pages, 
            and extract data using web crawling tools. When given a URL and a query:
            1. First, use the appropriate tool to fetch/crawl the website
            2. Extract relevant information based on the query
            3. Provide a clear, structured answer

            Think step by step and use the tools effectively.
            """
        )

        # Create user message with URL and query
        user_message = HumanMessage(
            content=f"""
            Please query this website: {url}
            Query: {query}
            Use the available tools to fetch the website content and answer the query.
            """
        )

        # Invoke agent
        messages = [system_message, user_message]
        print(messages)
        response = await self.agent.ainvoke({"messages": messages})

        print(response)

        # Extract agent response
        agent_response = response["messages"][-1].content

        result = {
            "url": url,
            "query": query,
            "response": agent_response,
        }

        # Add summary if requested
        if summarize:
            summary = await self.summarizer.summarize(
                content=agent_response,
                query=query,
            )
            result["summary"] = summary

        return result

    async def query_and_summarize(
        self,
        url: str,
        query: str,
    ) -> str:
        """
        Query a website and return summarized result.

        Args:
            url: URL of the website
            query: Query/question

        Returns:
            Summarized response
        """
        result = await self.query_website(url, query, summarize=True)
        return result.get("summary", result["response"])

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        if not self.tools:
            # Try to get tools synchronously
            try:
                self.tools = asyncio.run(self.mcp_manager.get_tools())
            except Exception:
                return []
        return [tool.name for tool in self.tools]

    async def interactive_mode(self):
        """Run agent in interactive mode."""
        await self._initialize_agent()

        print("=" * 70)
        print("Web Agent - Interactive Mode")
        print("=" * 70)
        print(f"Available Tools: {', '.join(self.get_available_tools())}")
        print("-" * 70)
        print("Enter 'quit' to exit")
        print("Format: <URL> | <QUERY>")
        print("=" * 70)

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if "|" in user_input:
                    parts = user_input.split("|", 1)
                    url = parts[0].strip()
                    query = parts[1].strip()
                else:
                    # Assume it's a URL, use default query
                    url = user_input
                    query = "What is this website about? Summarize the main content."

                print(f"\n🔍 Querying: {url}")
                print(f"❓ Query: {query}\n")

                result = await self.query_and_summarize(url, query)
                print("\n📝 Summary:")
                print("-" * 70)
                print(result)
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback

                traceback.print_exc()


async def main():
    web_agent = WebAgent()


async def test_agent():
    url = "https://www.google.com"
    query = "What is this website about?"
    system_message = SystemMessage(
        content="""You are a helpful web assistant that can scrape websites, crawl pages, 
and extract data using web crawling tools. When given a URL and a query:
1. First, use the appropriate tool to fetch/crawl the website
2. Extract relevant information based on the query
3. Provide a clear, structured answer

Think step by step and use the tools effectively."""
    )

    query_message = HumanMessage(
        content=f"""Please query this website: {url}
Query: {query}
Use the available tools to fetch the website content and answer the query."""
    )

    messages = [system_message, query_message]
    response = await LocalOllama().llm.ainvoke(messages)
    print(response)


if __name__ == "__main__":
    asyncio.run(test_agent())
