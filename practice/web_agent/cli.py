"""Command-line interface for Web Agent."""

import asyncio
import argparse
import sys
from web_agent.agent import WebAgent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Web Agent - Query websites and summarize information"
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="URL of the website to query",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query/question about the website",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model (default: 0.0)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't summarize results",
    )
    return parser.parse_args()


async def main():
    """Main CLI entry point."""
    args = parse_args()

    # Create agent
    agent = WebAgent(
        model_name=args.model,
        temperature=args.temperature,
    )

    # Interactive mode
    if args.interactive:
        await agent.interactive_mode()
        return

    # Check if URL and query provided
    if not args.url:
        print("Error: URL is required (or use --interactive mode)")
        sys.exit(1)

    if not args.query:
        args.query = "What is this website about? Summarize the main content."

    # Query website
    try:
        print(f"🔍 Querying: {args.url}")
        print(f"❓ Query: {args.query}\n")

        if args.no_summary:
            result = await agent.query_website(args.url, args.query, summarize=False)
            print("\n📄 Response:")
            print("-" * 70)
            print(result["response"])
        else:
            summary = await agent.query_and_summarize(args.url, args.query)
            print("\n📝 Summary:")
            print("-" * 70)
            print(summary)
            print("-" * 70)

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
