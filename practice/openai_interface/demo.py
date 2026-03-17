from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="llama3.2:3b",  # 或 gemma3:12b、llama3.2 等你拉取的 tag
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {
            "role": "user",
            "content": "Write a Python function to reverse a linked list.",
        },
    ],
    temperature=0.7,
    max_tokens=512,
    stream=False,  # 或 True 流式
)

print(response.choices[0].message.content)

# curl http://localhost:11434/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "llama3.2:3b",
#     "messages": [
#       {"role": "user", "content": "Hello, what models do you support?"}
#     ],
#     "stream": false
#   }'
