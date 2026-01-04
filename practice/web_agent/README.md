# Web Agent

A LangChain-based web agent that can query data from websites and summarize information using MCP (Model Context Protocol) tools for web crawling.

## Features

- 🔍 **Web Crawling**: Uses MCP tools (Firecrawl) to crawl and scrape websites
- 🤖 **LangChain Agent**: Built on LangGraph for intelligent web querying
- 📝 **Summarization**: Automatically summarizes extracted information
- 🔧 **MCP Integration**: Leverages Model Context Protocol for web operations

## Installation

1. Install dependencies:
```bash
pip install -e .
```

2. Install Node.js dependencies for MCP server:
```bash
npm install -g firecrawl-mcp
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Environment Variables

Create a `.env` file with the following:

```env
OPENAI_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key  # Optional, for Firecrawl MCP
```

## Usage

### Basic Usage

```python
from web_agent import WebAgent

agent = WebAgent()
result = await agent.query_and_summarize("https://example.com", "What is this website about?")
print(result)
```

### Command Line Interface

```bash
python -m web_agent.cli "https://example.com" "Summarize the main content"
```

### Interactive Mode

```bash
python -m web_agent.cli --interactive
```

## Project Structure

```
web_agent/
├── __init__.py
├── agent.py          # Main agent implementation
├── mcp_tools.py      # MCP tool integration
├── summarizer.py     # Summarization logic
└── cli.py           # Command-line interface
```

## How It Works

1. **Web Crawling**: Uses MCP Firecrawl tools to fetch and parse web pages
2. **Data Extraction**: Extracts relevant content from the crawled pages
3. **Agent Processing**: LangChain agent processes queries and extracts information
4. **Summarization**: Summarizes the extracted information using LLM

## Requirements

- Python 3.10+
- Node.js (for MCP server)
- OpenAI API key
- Firecrawl API key (optional, for advanced crawling)

