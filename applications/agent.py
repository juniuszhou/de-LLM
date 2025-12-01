import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import OpenAI
from langchain_community.llms import FakeListLLM

def run_agent():
    print("--- Simple LangChain Agent ---")
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("NOTICE: OPENAI_API_KEY not found. Using FakeListLLM for demonstration.")
        # Mock LLM that returns pre-determined responses to simulate the agent thinking
        llm = FakeListLLM(responses=[
            "I need to calculate 10 to the power of 3.", 
            "1000", 
            "The answer is 1000."
        ])
    else:
        print("Using OpenAI LLM.")
        llm = OpenAI(temperature=0)

    # Load tools
    # Note: llm-math requires an LLM to parse the question into a math problem
    try:
        tools = load_tools(["llm-math"], llm=llm)
    except Exception as e:
        print(f"Could not load standard tools: {e}")
        print("Proceeding without tools for demo.")
        tools = []

    # Initialize agent
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )

    # Run the agent
    query = "What is 10 raised to the power of 3?"
    print(f"\nQuery: {query}")
    try:
        response = agent.run(query)
        print(f"\nFinal Answer: {response}")
    except Exception as e:
        print(f"Error running agent: {e}")

if __name__ == "__main__":
    run_agent()