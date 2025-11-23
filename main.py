# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# text = "Hello world"
# tokens = tokenizer.tokenize(text)  # WordPiece 分词
# print(tokens)  # 输出: ['hello', ',', 'world', '!']

# # 转换为 ID
# inputs = tokenizer(text, return_tensors="pt")
# print(inputs['input_ids'])  # tensor([[ 101, 7592, 1010, 2088,  999,  102]])
# print(inputs['attention_mask'])

# text = "Hello, world! How are you?"
# inputs = tokenizer(text, return_tensors="pt")
# print(inputs['input_ids']) 
# print(inputs['attention_mask'])


import torch
import torch.nn as nn
from transformers import BertTokenizer

# 1. Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
texts = ["Hello world", "How are you, I am fine?"]
inputs = tokenizer(texts, padding=True, truncation=True, max_length=10, return_tensors="pt")
input_ids = inputs['input_ids']  # (2, 10)
print(input_ids)
attention_mask = inputs['attention_mask']  # (2, 10)
print(attention_mask)

# 2. Embedding
d_model = 768  # BERT-base 的 d_model
vocab_size = tokenizer.vocab_size
print(vocab_size)
embedding = nn.Embedding(vocab_size, d_model)
embedded = embedding(input_ids)

print(embedded.shape)