from typing import TypedDict, Dict, Optional
from langgraph.graph import StateGraph, START, END

# define a state
class AgentState(TypedDict):
    message: str 
    context: str
    history: Optional[str]

# define a node
def greeting_node(state: AgentState) -> AgentState:
    """Greeting node"""
    state["message"] = "Hello, how are you?"
    state["context"] = "Hello, my name is John"
    state["history"] = "History added into state"
    return state


graph = StateGraph(AgentState)
graph.add_node("greeting", greeting_node)

# start and end point
# graph.set_entry_point("greeting")
graph.add_edge(START, "greeting")
# graph.set_finish_point("greeting")
graph.add_edge("greeting", END)

app = graph.compile()

result = app.invoke({"message1": "Hello, my name is John"})
print(result["message"])
print(result["context"])
print(result["history"])
