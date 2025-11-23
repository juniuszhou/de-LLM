from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Hello, world!"
tokens = tokenizer.tokenize(text)  # WordPiece 分词
print(tokens)  # 输出: ['hello', ',', 'world', '!']

# 转换为 ID
inputs = tokenizer(text, return_tensors="pt")
print(inputs['input_ids'])  # tensor([[ 101, 7592, 1010, 2088,  999,  102]])


text = "Hello, world! How are you?"
inputs = tokenizer(text, return_tensors="pt")
print(inputs['input_ids']) 