"""Summarization functionality for web content."""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


class Summarizer:
    """Handles summarization of web content."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize summarizer.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for model generation
            api_key: OpenAI API key (if None, reads from env)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
        self.output_parser = StrOutputParser()

    def create_summarize_prompt(
        self, query: Optional[str] = None
    ) -> ChatPromptTemplate:
        """
        Create a prompt template for summarization.

        Args:
            query: Optional query to focus the summary

        Returns:
            ChatPromptTemplate for summarization
        """
        if query:
            template = """You are a helpful assistant that summarizes web content.

Given the following web content, provide a concise summary that addresses this query: {query}

Web Content:
{content}

Please provide a clear, well-structured summary that answers the query."""
        else:
            template = """You are a helpful assistant that summarizes web content.

Given the following web content, provide a concise summary covering the main points.

Web Content:
{content}

Please provide a clear, well-structured summary."""
        return ChatPromptTemplate.from_template(template)

    async def summarize(
        self,
        content: str,
        query: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Summarize web content.

        Args:
            content: The web content to summarize
            query: Optional query to focus the summary
            max_length: Optional maximum length for summary

        Returns:
            Summarized content
        """
        prompt = self.create_summarize_prompt(query)
        chain = prompt | self.llm | self.output_parser

        summary = await chain.ainvoke({"content": content, "query": query or ""})

        if max_length and len(summary) > max_length:
            # Truncate if needed
            summary = summary[:max_length] + "..."

        return summary

    def summarize_sync(
        self,
        content: str,
        query: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Synchronous version of summarize.

        Args:
            content: The web content to summarize
            query: Optional query to focus the summary
            max_length: Optional maximum length for summary

        Returns:
            Summarized content
        """
        import asyncio

        return asyncio.run(self.summarize(content, query, max_length))
