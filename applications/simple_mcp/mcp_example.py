from fastapi import FastAPI, Query
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel

"""
curl -s -X POST http://127.0.0.1:8000/add \
  -H "Content-Type: application/json" \
  -d '{"a": 42, "b": 58}' | jq
"""

app = FastAPI(
    title="Simple MCP Service",
    description="A minimal FastAPI app exposed as an MCP server for AI agents",
    version="1.0.0",
)


# Example Pydantic model for structured input
class AddRequest(BaseModel):
    a: int
    b: int


# Simple endpoints that will automatically become MCP tools
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Simple MCP Service is running!"}


@app.get("/hello")
async def hello(name: str = Query("World", description="Name to greet")):
    """Greet someone"""
    return {"message": f"Hello, {name}!"}


@app.post("/add")
async def add_numbers(request: AddRequest):
    """Add two numbers"""
    return {"result": request.a + request.b}


@app.get("/echo")
async def echo(text: str):
    """Echo back the input text"""
    return {"echo": text}


# === MCP Integration (this is the magic) ===
mcp = FastApiMCP(
    app,
    name="Simple MCP Service",
    description="A simple demonstration of FastAPI exposed as MCP tools for AI agents",
)

# Mount the MCP server at /mcp (recommended path)
mcp.mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
