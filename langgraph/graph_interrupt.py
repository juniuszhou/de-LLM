from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END, START
from langgraph.types import Interrupt, interrupt, Send

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

class AgentState(TypedDict):
    messages: str 

def start_agent(state: AgentState) -> AgentState:
    return {"messages": state["messages"] + "Hello, how are you?"}

def interrupt_handler(state: AgentState) -> AgentState:
    approved = interrupt({"error": "what is your problem?"})
    return {"messages": approved}

graph = (
    StateGraph(AgentState)
    .add_node("start_agent", start_agent)
    .add_node("interrupt_handler", interrupt_handler)
    .add_edge(START, "start_agent")
    .add_edge("start_agent", "interrupt_handler")
    .add_edge("interrupt_handler", END)
    .compile()
)

# ============================================================================
# WHY IS __interrupt__ A LIST/TUPLE?
# ============================================================================
# 
# Question: If the graph stops at the first interrupt, why return a list?
#
# Answer: The list structure exists because:
#
# 1. PARALLEL EXECUTION: Multiple nodes can run concurrently (in parallel)
#    - When nodes execute in parallel, multiple interrupts can occur simultaneously
#    - Each parallel node can independently call interrupt()
#    - All interrupts from that execution step are collected in the list
#
# 2. API CONSISTENCY: Always return the same structure
#    - Whether 0, 1, or multiple interrupts occur, the structure is consistent
#    - Makes parsing code simpler and more predictable
#
# 3. SEQUENTIAL EXECUTION: In sequential graphs, you'll typically get 1 element
#    - Execution stops at the first interrupt encountered
#    - The list will contain exactly 1 Interrupt object
#    - But the API still returns a list for consistency
#
# 4. FUTURE-PROOFING: Supports advanced use cases
#    - Subgraphs, parallel branches, conditional execution
#    - Multiple tasks executing in the same step
#
# ============================================================================

# ============================================================================
# Example 1: Single interrupt (original example)
# ============================================================================
print("=" * 70)
print("Example 1: Single Interrupt (Sequential Execution)")
print("=" * 70)
print("Note: In sequential execution, only ONE interrupt occurs.")
print("      The list will contain 1 element.\n")

result = graph.invoke({"messages": "I am junius."})

# output [Interrupt(value='what is your problem?', id='c3ac02816a39cfc1dabcbd1d8080b8e0')]
interrupt_list = result["__interrupt__"]
print(f"Interrupt list: {interrupt_list}")
print(f"Type: {type(interrupt_list)}")
print(f"Length: {len(interrupt_list)}")

# Parse the interrupt message: it's a list containing Interrupt objects
if interrupt_list:
    interrupt_message = interrupt_list[0]  # Get the first Interrupt object
    print(f"\nInterrupt value: {interrupt_message.value}")
    print(f"Interrupt id: {interrupt_message.id}")

# ============================================================================
# Example 2: Sequential nodes - only first interrupt occurs
# ============================================================================
# In sequential execution, only the FIRST interrupt stops execution.
# Subsequent nodes won't run until you resume.
print("\n" + "=" * 70)
print("Example 2: Sequential Nodes (Only First Interrupt Occurs)")
print("=" * 70)
print("Note: Execution stops at node1. Node2 and node3 never execute.")
print("      You'll only see 1 interrupt in the list.\n")

def node1_handler(state: AgentState) -> AgentState:
    """First node that creates an interrupt"""
    approved = interrupt({"type": "approval", "message": "Do you approve step 1?"})
    return {"messages": state["messages"] + f" | Step 1 approved: {approved}"}

def node2_handler(state: AgentState) -> AgentState:
    """Second node that creates an interrupt"""
    approved = interrupt({"type": "verification", "message": "Please verify step 2"})
    return {"messages": state["messages"] + f" | Step 2 verified: {approved}"}

def node3_handler(state: AgentState) -> AgentState:
    """Third node that creates an interrupt"""
    approved = interrupt({"type": "confirmation", "message": "Confirm step 3?"})
    return {"messages": state["messages"] + f" | Step 3 confirmed: {approved}"}

# Create a graph with multiple nodes that can interrupt
multi_interrupt_graph = (
    StateGraph(AgentState)
    .add_node("node1", node1_handler)
    .add_node("node2", node2_handler)
    .add_node("node3", node3_handler)
    .add_edge(START, "node1")
    .add_edge("node1", "node2")
    .add_edge("node2", "node3")
    .add_edge("node3", END)
    .compile()
)

# When multiple nodes interrupt, you'll get multiple interrupts
# Note: In practice, execution stops at the first interrupt
result2 = multi_interrupt_graph.invoke({"messages": "Starting multi-step process."})
interrupt_list2 = result2.get("__interrupt__", [])

print(f"Number of interrupts: {len(interrupt_list2)}")
for i, interrupt_obj in enumerate(interrupt_list2):
    print(f"\nInterrupt {i+1}:")
    print(f"  Type: {interrupt_obj.value.get('type', 'unknown')}")
    print(f"  Message: {interrupt_obj.value.get('message', 'N/A')}")
    print(f"  ID: {interrupt_obj.id}")

# ============================================================================
# Example 3: Multiple interrupts in a single node
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Multiple Interrupts in Single Node")
print("=" * 70)

def multi_interrupt_node(state: AgentState) -> AgentState:
    """A node that creates multiple interrupts sequentially"""
    # First interrupt - approval
    approval = interrupt({"step": 1, "action": "approve", "message": "Approve this action?"})
    
    # Second interrupt - verification  
    verification = interrupt({"step": 2, "action": "verify", "message": "Verify the details?"})
    
    # Third interrupt - confirmation
    confirmation = interrupt({"step": 3, "action": "confirm", "message": "Confirm completion?"})
    
    return {
        "messages": state["messages"] + f" | Approved: {approval}, Verified: {verification}, Confirmed: {confirmation}"
    }

single_node_multi_interrupt_graph = (
    StateGraph(AgentState)
    .add_node("multi_interrupt", multi_interrupt_node)
    .add_edge(START, "multi_interrupt")
    .add_edge("multi_interrupt", END)
    .compile()
)

# When a node has multiple interrupts, execution stops at the first one
# To continue, you need to resume with Command for each interrupt in order
result3 = single_node_multi_interrupt_graph.invoke({"messages": "Testing multiple interrupts."})
interrupt_list3 = result3.get("__interrupt__", [])

print(f"Number of interrupts: {len(interrupt_list3)}")
if interrupt_list3:
    print(f"\nFirst interrupt (execution stops here):")
    print(f"  Step: {interrupt_list3[0].value.get('step', 'N/A')}")
    print(f"  Action: {interrupt_list3[0].value.get('action', 'N/A')}")
    print(f"  Message: {interrupt_list3[0].value.get('message', 'N/A')}")
    print(f"  ID: {interrupt_list3[0].id}")
    print("\nNote: To continue, resume with Command. The node will re-execute")
    print("      and stop at the next interrupt until all are resolved.")

# ============================================================================
# Example 4: Parsing multiple interrupts
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: Parsing Multiple Interrupts")
print("=" * 70)

def parse_interrupts(interrupt_list):
    """Helper function to parse a list of interrupts"""
    if not interrupt_list:
        print("No interrupts found.")
        return
    
    print(f"Found {len(interrupt_list)} interrupt(s):\n")
    
    for idx, interrupt_obj in enumerate(interrupt_list, 1):
        print(f"Interrupt #{idx}:")
        print(f"  ID: {interrupt_obj.id}")
        
        # Handle different value types
        value = interrupt_obj.value
        if isinstance(value, dict):
            print(f"  Value (dict):")
            for key, val in value.items():
                print(f"    {key}: {val}")
        else:
            print(f"  Value: {value}")
        print()

# Test parsing with the interrupts we collected
print("Parsing interrupts from Example 2:")
parse_interrupts(interrupt_list2)

print("Parsing interrupts from Example 3:")
parse_interrupts(interrupt_list3)

# ============================================================================
# Example 5: When Multiple Interrupts CAN Actually Occur
# ============================================================================
# Multiple interrupts occur when nodes run in PARALLEL (concurrent execution)
# This happens with conditional edges that trigger multiple nodes simultaneously,
# or when using Send() to invoke multiple nodes in parallel.
print("\n" + "=" * 70)
print("Example 5: When Multiple Interrupts Actually Occur")
print("=" * 70)
print("""
Multiple interrupts occur when:
1. Multiple nodes execute in PARALLEL (same execution step)
2. Conditional edges trigger multiple branches simultaneously  
3. Using Send() to invoke multiple nodes concurrently

In these cases, each parallel node can independently call interrupt(),
and ALL interrupts from that step are collected in the list.

However, note that:
- Sequential execution: Only 1 interrupt (first one encountered)
- Parallel execution: Multiple interrupts possible (all from same step)
- The list structure handles both cases uniformly
""")

# Example showing the concept (would need conditional edges or Send for real parallel execution)
def parallel_node_a(state: AgentState) -> AgentState:
    """Node A that interrupts"""
    approved = interrupt({"node": "A", "message": "Approve from node A?"})
    return {"messages": state["messages"] + f" | Node A: {approved}"}

def parallel_node_b(state: AgentState) -> AgentState:
    """Node B that interrupts"""
    approved = interrupt({"node": "B", "message": "Approve from node B?"})
    return {"messages": state["messages"] + f" | Node B: {approved}"}

def parallel_node_c(state: AgentState) -> AgentState:
    """Node C that interrupts"""
    approved = interrupt({"node": "C", "message": "Approve from node C?"})
    return {"messages": state["messages"] + f" | Node C: {approved}"}

# Note: To actually get multiple interrupts, you'd need to configure
# the graph to run these nodes in parallel using conditional edges
# or Send(). In sequential execution, only the first will interrupt.

print("""
Summary:
--------
- result["__interrupt__"] is ALWAYS a list/tuple (even if empty or single element)
- Sequential execution: List contains 1 interrupt (execution stops at first)
- Parallel execution: List can contain multiple interrupts (from same step)
- Always check len(interrupt_list) and iterate if needed
- The list structure provides API consistency and supports parallel execution
""")

# ============================================================================
# Example 6: TRUE PARALLEL EXECUTION using Send() and conditional edges
# ============================================================================
# This is the CORRECT way to run nodes in parallel and get multiple interrupts
print("\n" + "=" * 70)
print("Example 6: TRUE Parallel Execution with Send() and Conditional Edges")
print("=" * 70)

# Define nodes that will run in parallel
def parallel_node_a(state: AgentState) -> AgentState:
    """Node A that interrupts - runs in parallel"""
    approved = interrupt({"node": "A", "message": "Approve from node A?"})
    return {"messages": state["messages"] + f" | Node A: {approved}"}

def parallel_node_b(state: AgentState) -> AgentState:
    """Node B that interrupts - runs in parallel"""
    approved = interrupt({"node": "B", "message": "Approve from node B?"})
    return {"messages": state["messages"] + f" | Node B: {approved}"}

def parallel_node_c(state: AgentState) -> AgentState:
    """Node C that interrupts - runs in parallel"""
    approved = interrupt({"node": "C", "message": "Approve from node C?"})
    return {"messages": state["messages"] + f" | Node C: {approved}"}

def distributor(state: AgentState):
    """Distributor function that sends to multiple nodes in parallel"""
    # Return a list of Send objects - these nodes will execute IN PARALLEL
    return [
        Send("parallel_node_a", {"messages": state["messages"] + " [A]"}),
        Send("parallel_node_b", {"messages": state["messages"] + " [B]"}),
        Send("parallel_node_c", {"messages": state["messages"] + " [C]"}),
    ]

def aggregator(state: AgentState) -> AgentState:
    """Aggregator to collect results (won't run if interrupts occur)"""
    return {"messages": state["messages"] + " | Aggregated"}

# Create graph with TRUE parallel execution
# Key: Use add_conditional_edges with a function that returns a LIST of Send objects
parallel_graph = (
    StateGraph(AgentState)
    .add_node("parallel_node_a", parallel_node_a)
    .add_node("parallel_node_b", parallel_node_b)
    .add_node("parallel_node_c", parallel_node_c)
    .add_node("aggregator", aggregator)
    # Conditional edge from START returns list of Send objects - nodes run in parallel
    .add_conditional_edges(START, distributor)  # Returns [Send(...), Send(...), Send(...)]
    .add_edge("parallel_node_a", "aggregator")
    .add_edge("parallel_node_b", "aggregator")
    .add_edge("parallel_node_c", "aggregator")
    .add_edge("aggregator", END)
    .compile()
)

print("\nRunning parallel execution...")
print("All three nodes (A, B, C) should execute in parallel.")
print("If all interrupt, you should see MULTIPLE interrupts in the list.\n")

result4 = parallel_graph.invoke({"messages": "Testing TRUE parallel execution."})
interrupt_list4 = result4.get("__interrupt__", [])

print(f"Number of interrupts: {len(interrupt_list4)}")
if len(interrupt_list4) > 1:
    print("✅ SUCCESS! Multiple interrupts occurred (parallel execution confirmed)")
else:
    print("⚠️  Only 1 interrupt (execution may have stopped at first interrupt)")

for i, interrupt_obj in enumerate(interrupt_list4):
    print(f"\nInterrupt {i+1}:")
    print(f"  Node: {interrupt_obj.value.get('node', 'N/A')}")
    print(f"  Message: {interrupt_obj.value.get('message', 'N/A')}")
    print(f"  ID: {interrupt_obj.id}")

# ============================================================================
# Example 7: Simplified parallel execution example
# ============================================================================
print("\n" + "=" * 70)
print("Example 7: Simplified Parallel Execution Pattern")
print("=" * 70)

def simple_parallel_node(state: AgentState) -> AgentState:
    """A node that interrupts - will be called multiple times in parallel"""
    # Extract node identifier from messages (we encode it in the distributor)
    if "[Node1]" in state["messages"]:
        node_name = "Node1"
    elif "[Node2]" in state["messages"]:
        node_name = "Node2"
    elif "[Node3]" in state["messages"]:
        node_name = "Node3"
    else:
        node_name = "Unknown"
    
    approved = interrupt({"node": node_name, "message": f"Approve from {node_name}?"})
    return {"messages": state["messages"] + f" | {node_name}: {approved}"}

def simple_distributor(state: AgentState):
    """Distribute to multiple nodes in parallel"""
    # Send same state to multiple nodes - they'll run in parallel
    # We'll differentiate them by the order they're called
    return [
        Send("simple_parallel_node", {"messages": state["messages"] + " [Node1]"}),
        Send("simple_parallel_node", {"messages": state["messages"] + " [Node2]"}),
        Send("simple_parallel_node", {"messages": state["messages"] + " [Node3]"}),
    ]

simple_parallel_graph = (
    StateGraph(AgentState)
    .add_node("simple_parallel_node", simple_parallel_node)
    # Conditional edge directly returns list of Send objects - nodes run in parallel
    .add_conditional_edges(START, simple_distributor)  # Returns [Send(...), Send(...), Send(...)]
    .add_edge("simple_parallel_node", END)
    .compile()
)

print("\nRunning simplified parallel execution...")
result5 = simple_parallel_graph.invoke({"messages": "Simple parallel test."})
interrupt_list5 = result5.get("__interrupt__", [])

print(f"\nNumber of interrupts: {len(interrupt_list5)}")
for i, interrupt_obj in enumerate(interrupt_list5):
    print(f"  Interrupt {i+1}: {interrupt_obj.value.get('node', 'N/A')}")

print("""
Key Takeaways:
--------------
1. Use add_conditional_edges() with a function that returns a LIST of Send() objects
2. Each Send() object targets a different node with potentially different state
3. All nodes specified in the Send() list execute IN PARALLEL
4. If multiple parallel nodes call interrupt(), ALL interrupts are collected
5. The result["__interrupt__"] list will contain multiple Interrupt objects
6. This is the ONLY way to get multiple interrupts in a single execution step
""")
