# Quick Start Guide

## Prerequisites

1. **Python 3.10+** installed
2. **Node.js** installed (for MCP server)
3. **OpenAI API Key** (required)
4. **Firecrawl API Key** (optional, for advanced crawling)

## Installation Steps

### 1. Install Python Dependencies

```bash
cd practice/web_agent
pip install -r requirements.txt
# OR
pip install -e .
```

### 2. Install MCP Server (Firecrawl)

```bash
npm install -g firecrawl-mcp
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-your-openai-key-here
FIRECRAWL_API_KEY=your-firecrawl-key-here  # Optional
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.0
```

## Usage Examples

### Python Script

```python
import asyncio
from web_agent import WebAgent

async def main():
    agent = WebAgent()
    result = await agent.query_and_summarize(
        url="https://www.example.com",
        query="What is this website about?",
    )
    print(result)

asyncio.run(main())
```

### Command Line - Single Query

```bash
python -m web_agent.cli "https://www.example.com" "What is this website about?"
```

### Command Line - Interactive Mode

```bash
python -m web_agent.cli --interactive
```

Then enter queries in the format:

```
https://www.example.com | What is this website about?
```

### Run Example Script

```bash
python example.py
```

## How It Works

1. **MCP Integration**: The agent uses Model Context Protocol (MCP) to connect to Firecrawl server
2. **Web Crawling**: Firecrawl MCP tools fetch and parse web pages
3. **Agent Processing**: LangGraph agent processes the query and extracts relevant information
4. **Summarization**: LLM summarizes the extracted content

## Troubleshooting

### MCP Server Not Found

If you get an error about MCP server:

```bash
npm install -g firecrawl-mcp
```

### Missing API Keys

Make sure your `.env` file has:

- `OPENAI_API_KEY` (required)
- `FIRECRAWL_API_KEY` (optional but recommended)

### Import Errors

Make sure you're in the project directory and dependencies are installed:

```bash
pip install -r requirements.txt
```

## Next Steps

- Check `README.md` for detailed documentation
- See `example.py` for more usage examples
- Customize the agent in `web_agent/agent.py`
