from typing import TypedDict, Dict, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# define a state
class AgentState(TypedDict):
    message: List[HumanMessage]
    

# Initialize LLM with local Ollama service
llm: BaseChatModel = ChatOpenAI(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# define a node
def call_llm_node(state: AgentState) -> AgentState:
    """Greeting node"""
    response = llm.invoke(state["message"])
    print(response.content)
    return state


graph = StateGraph(AgentState)
graph.add_node("llm_node", call_llm_node)

# start and end point
# graph.set_entry_point("greeting")
graph.add_edge(START, "llm_node")
# graph.set_finish_point("greeting")
graph.add_edge("llm_node", END)

app = graph.compile()

result = app.invoke({"message": "Hello, my name is John"})

