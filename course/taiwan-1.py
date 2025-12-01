from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.login_huggingface import login_huggingface

def main():
    login_huggingface()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "which city is capital of singapore？ answer"},
    ]

    tokenizer.apply_chat_template(messages)

    prompt = "which city is capital of singapore？ answer" 
    print("輸入的 prompt 是:", prompt)

    basic_info(tokenizer, model)

    # next_token(tokenizer, model, prompt)
    # generate(tokenizer, model, prompt, False)

def basic_info(tokenizer, model):
    length = tokenizer.vocab_size
    print("詞表大小是:", length)
    last_token = tokenizer.decode(length-1)
    print("最後一個 token 是:", last_token)

    token = "hello"
    token_id = tokenizer.encode(token, add_special_tokens=False)
    print("hello's token id 是:", token_id)
    original_token = tokenizer.decode(token_id)
    print("hello's original token 是:", original_token)

# continuously generate next token
def next_token(tokenizer, model, prompt, max_length=40):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print("這是 model 可以讀的輸入：",input_ids)

    outputs = model(input_ids)
    last_logits = outputs.logits[:, -1, :] 
    probabilities = torch.softmax(last_logits, dim=-1) 

    top_k = 10
    top_p, top_indices = torch.topk(probabilities, top_k)
    for i in range(top_k):
        token_id = top_indices[0][i].item()
        probability = top_p[0][i].item()
        token_str = tokenizer.decode(token_id)
        print(f"Token ID: {token_id}, Token: '{token_str}', 機率: {probability:.4f}")
        prompt += token_str
        print("更新後的 prompt 是:", prompt)
    

# use generate to get a response with system prompt
def generate(tokenizer, model, prompt, system_prompt, max_length=400):
    if system_prompt: 
        prompt = "使用者說：" + prompt + "\nAI回答："

    messages = [    
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # input_ids = tokenizer.encode(prompt, return_tensors="pt", )
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=3, pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids))

    # output = output.filter(lambda x: x != tokenizer.eos_token_id)
    result = tokenizer.decode(output[0], skip_special_tokens=False)
    print("生成的結果是:", result)
    response = result.split("<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip() #把 AI 的回答取出

    print("生成的答案是:", response)

if __name__ == "__main__":
    main()