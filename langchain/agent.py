"""
LangChain Agent with Local LLM (Ollama)
This script creates a working agent that uses a local LLM via OpenAI-compatible interface.
Works without langchain.agents module by using direct LLM tool calling.

Installation Requirements:
    pip install langchain langchain-openai langchain-core

This script does NOT require:
    - langchain.agents (uses custom SimpleAgent instead)
    - langchain-community (unless you want additional tools)

Usage:
    1. Make sure Ollama is running: ollama serve
    2. Pull the model: ollama pull llama3.2:3b
    3. Run: python langchain/agent.py
"""

import os
import json

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("\nPlease install:")
    print("  pip install langchain langchain-openai langchain-core")
    print("\nOr using uv:")
    print("  uv add langchain langchain-openai langchain-core")
    raise

# Set up local LLM (Ollama endpoint)
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy key; Ollama doesn't need a real one

# Initialize LLM with local Ollama service
llm = ChatOpenAI(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

hasattr(llm, 'bind_tools')

# Define tools for the agent (as regular functions)
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Safe evaluation - only allow basic math operations
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current date and time as a string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city: str) -> str:
    """Get weather information for a city (mock implementation).
    
    Args:
        city: Name of the city
    
    Returns:
        Mock weather information
    """
    # This is a mock function - in production, you'd call a real weather API
    return f"The weather in {city} is sunny, 22°C with light winds."

# Tool definitions for the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression safely. Input should be a string like '2 + 2' or '10 * 5'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Tool function mapping
TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "get_weather": get_weather
}

class SimpleAgent:
    """Simple agent implementation without langchain.agents module"""
    
    def __init__(self, llm, tools_config, tool_functions, verbose=True):
        self.llm = llm
        # self.tools_config = tools_config
        # self.tool_functions = tool_functions
        self.tools_config = []
        self.tool_functions = {}
        self.verbose = verbose
        self.conversation_history = []
        
    def invoke(self, input_data):
        """Invoke the agent with input"""
        query = input_data.get("input", input_data.get("messages", [])[-1].content if isinstance(input_data.get("messages"), list) else "")
        
        if isinstance(query, str):
            messages = [HumanMessage(content=query)]
        else:
            messages = input_data.get("messages", [HumanMessage(content=query)])
        
        # Add system message
        system_msg = SystemMessage(content="""You are a helpful AI assistant that can use tools to answer questions.
        When you need to perform calculations, get the current time, or check weather, use the appropriate tools.
        Always be accurate and helpful in your responses.""")
        
        # Try to bind tools to LLM (if supported)
        try:
            if hasattr(self.llm, 'bind_tools'):
                llm_with_tools = self.llm.bind_tools(self.tools_config)
            else:
                llm_with_tools = self.llm
                if self.verbose:
                    print("Note: LLM doesn't support bind_tools, using basic mode")
        except Exception as e:
            llm_with_tools = self.llm
            if self.verbose:
                print(f"Tool binding not available: {e}, using basic mode")
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Prepare messages with tool descriptions in system message
            tool_descriptions = "\n".join([
                f"- {tool['function']['name']}: {tool['function']['description']}"
                for tool in self.tools_config
            ])
            
            enhanced_system_msg = SystemMessage(content=f"""You are a helpful AI assistant that can use tools to answer questions.

Available tools:
{tool_descriptions}

When you need to use a tool, respond in this format:
TOOL: tool_name
ARGS: {{"arg1": "value1"}}

For example:
TOOL: calculate
ARGS: {{"expression": "2 + 2"}}

Always be accurate and helpful in your responses.""")
            
            full_messages = [enhanced_system_msg] + self.conversation_history + messages
            
            if self.verbose:
                print(f"\n[Iteration {iteration}] Calling LLM...")
            
            # Call LLM
            try:
                response = llm_with_tools.invoke(full_messages)
            except Exception as e:
                if self.verbose:
                    print(f"LLM call failed: {e}")
                response = self.llm.invoke(full_messages)
            
            # Add response to conversation
            self.conversation_history.extend(messages)
            self.conversation_history.append(response)
            
            # Get response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Check if LLM wants to call a tool (multiple methods)
            tool_calls = None
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = response.tool_calls
            elif 'TOOL:' in response_content:
                # Parse tool call from text response
                import re
                tool_match = re.search(r'TOOL:\s*(\w+)', response_content)
                args_match = re.search(r'ARGS:\s*(\{.*?\})', response_content, re.DOTALL)
                if tool_match:
                    tool_calls = [{
                        'name': tool_match.group(1),
                        'args': json.loads(args_match.group(1)) if args_match else {}
                    }]
            
            if tool_calls:
                if self.verbose:
                    print(f"[Tool Calls] {len(tool_calls)} tool(s) requested")
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('name', tool_call.get('function', {}).get('name', ''))
                        tool_args = tool_call.get('args', tool_call.get('function', {}).get('arguments', {}))
                    else:
                        # Handle different tool call formats
                        tool_name = getattr(tool_call, 'name', '')
                        tool_args = getattr(tool_call, 'args', {})
                    
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            tool_args = {}
                    
                    if self.verbose:
                        print(f"  → Calling tool: {tool_name} with args: {tool_args}")
                    
                    if tool_name in self.tool_functions:
                        try:
                            result = self.tool_functions[tool_name](**tool_args)
                            tool_results.append(f"Tool {tool_name} returned: {result}")
                            if self.verbose:
                                print(f"====  ✓ Result: {result}")
                        except Exception as e:
                            tool_results.append(f"Tool {tool_name} error: {str(e)}")
                            if self.verbose:
                                print(f"  ✗ Error: {e}")
                    else:
                        tool_results.append(f"Unknown tool: {tool_name}")
                
                # Add tool results to messages for next iteration
                messages = [AIMessage(content="\n".join(tool_results))]
                continue
            else:
                # No tool calls, return the response
                return {"output": response_content}
        
        # Max iterations reached
        final_answer = response.content if hasattr(response, 'content') else str(response)
        return {"output": final_answer}

# Create the agent
try:
    # Try to use tool binding (requires LLM to support it)
    agent_executor = SimpleAgent(llm, TOOLS, TOOL_FUNCTIONS, verbose=True)
except Exception as e:
    print(f"Warning: Could not create agent with tools: {e}")
    agent_executor = None

def main():
    """Main function to run the agent."""
    print("=" * 70)
    print("LangChain Agent with Local LLM (Ollama)")
    print("=" * 70)
    print(f"Using model: llama3.2:3b")
    print(f"Ollama endpoint: http://localhost:11434/v1")
    print("=" * 70)
    
    # Test queries
    test_queries = []
    #     "What is the capital of France?",
    #     "What is 25 multiplied by 4?",
    #     "What time is it now?",
    #     "What's the weather like in Paris?",
    # ]
    
    if agent_executor is None:
        print("❌ Agent executor not available.")
        print("Trying fallback: direct LLM without tools...")
        # Fallback: just use LLM directly
        for query in test_queries:
            print(f"\n🤔 Query: {query}")
            print("-" * 70)
            try:
                response = llm.invoke([HumanMessage(content=query)])
                answer = response.content if hasattr(response, 'content') else str(response)
                print(f"✅ Answer: {answer}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        return
    
    for query in test_queries:
        print(f"\n🤔 Query: {query}")
        print("-" * 70)
        try:
            result = agent_executor.invoke({"input": query})
            print(f"✅ Answer: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        print("-" * 70)
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("Interactive Mode (type 'exit' to quit)")
    print("=" * 70)
    
    if agent_executor is None:
        return
        
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            result = agent_executor.invoke({"input": user_input})
            print(f"Agent: {result['output']}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()