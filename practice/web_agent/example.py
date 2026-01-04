"""Example usage of Web Agent."""

import asyncio
from web_agent import WebAgent


async def example_basic():
    """Basic example of querying a website."""
    print("=" * 70)
    print("Example 1: Basic Website Query")
    print("=" * 70)

    agent = WebAgent()

    result = await agent.query_and_summarize(
        url="https://www.example.com",
        query="What is this website about?",
    )

    print("\nResult:")
    print(result)


async def example_detailed():
    """Detailed example with full response."""
    print("\n" + "=" * 70)
    print("Example 2: Detailed Query with Full Response")
    print("=" * 70)

    agent = WebAgent()

    result = await agent.query_website(
        url="https://www.example.com",
        query="What information does this website provide?",
        summarize=True,
    )

    print(f"\nURL: {result['url']}")
    print(f"Query: {result['query']}")
    print(f"\nFull Response:")
    print("-" * 70)
    print(result["response"])
    print("-" * 70)
    print(f"\nSummary:")
    print("-" * 70)
    print(result["summary"])


async def example_custom_query():
    """Example with custom query."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Query")
    print("=" * 70)

    agent = WebAgent()

    result = await agent.query_and_summarize(
        url="https://www.python.org",
        query="What is Python programming language? What are its main features?",
    )

    print("\nResult:")
    print(result)


async def main():
    """Run all examples."""
    try:
        await example_basic()
        await example_detailed()
        # Uncomment to run more examples
        # await example_custom_query()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
