from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    dtype="auto",
    trust_remote_code=False,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

from transformers import pipeline

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

while True: 
    print("=" * 50)
    messages = [
        {"role": "user", "content": input("Enter your message: ")}
    ]
    print(messages)
    output = generator(messages)
    print(output[0]["generated_text"])

# Generate output
# output = generator(messages)
# print(output[0]["generated_text"])

