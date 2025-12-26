from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

agent = create_agent(model=llm)

result = agent.invoke(
    {
        "input": "What is the capital of France?",
        "messages": [HumanMessage(content="What is the capital of France?")],
    }
)
print(result["messages"])
print("="*50)
results = agent.batch(
    [
        {
            "input": "What is the capital of France?",
            "messages": [HumanMessage(content="What is the capital of France?")],
        },
        {
            "input": "What is the capital of France?",
            "messages": [HumanMessage(content="What is the capital of France?")],
        },
    ]
)
for result in results:
    print(result["messages"])
print("="*50)

results = agent.stream(
    {
            "input": "What is the capital of France?",
            "messages": [HumanMessage(content="What is the capital of France?")],
        },
)
for result in results:
    print(result)


