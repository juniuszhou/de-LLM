import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # 初始化位置编码
        pos_encoding = torch.zeros(1, max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_encoding[0, pos, i] = math.sin(pos / 10000 ** (2 * i / d_model))
                if i + 1 < d_model:  # 防止索引越界
                    pos_encoding[0, pos, i+1] = math.cos(pos / 10000 ** (2 * i / d_model))
        
        # 将初始化好的位置编码注册为参数
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        embed = self.embedding(x) * math.sqrt(self.d_model)  # 缩放嵌入
        embed = embed + self.pos_encoding[:, :x.size(1), :]
        return embed

# 示例：使用500个词汇的词嵌入
print("=== WordEmbedding 示例 ===")

# 参数设置
vocab_size = 500  # 词汇表大小
d_model = 512     # 词嵌入维度
max_len = 100     # 最大序列长度

# 创建模型
model = WordEmbedding(vocab_size, d_model, max_len)
print(f"模型参数: vocab_size={vocab_size}, d_model={d_model}, max_len={max_len}")

# 示例1: 单个句子
print("\n--- 示例1: 单个句子 ---")
sentence_tokens = torch.tensor([[1, 45, 123, 67, 234, 89, 2]])  # [1, 7] - 1个句子，7个token
embeddings_single = model(sentence_tokens)
print(f"输入形状: {sentence_tokens.shape}")
print(f"输出形状: {embeddings_single.shape}")
print(f"第一个token的嵌入向量前5维: {embeddings_single[0, 0, :5]}")

# 示例2: 批量处理
print("\n--- 示例2: 批量处理 ---")
batch_size = 8
seq_len = 15
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机生成token ids
embeddings_batch = model(input_ids)
print(f"批量输入形状: {input_ids.shape}")
print(f"批量输出形状: {embeddings_batch.shape}")

# 示例3: 不同长度的序列
print("\n--- 示例3: 不同长度序列 ---")
short_seq = torch.randint(0, vocab_size, (1, 5))   # 短序列
long_seq = torch.randint(0, vocab_size, (1, 50))   # 长序列

short_embeddings = model(short_seq)
long_embeddings = model(long_seq)

print(f"短序列 {short_seq.shape} -> {short_embeddings.shape}")
print(f"长序列 {long_seq.shape} -> {long_embeddings.shape}")

# 示例4: 展示位置编码的影响
print("\n--- 示例4: 位置编码影响 ---")
same_token = torch.tensor([[100, 100, 100]])  # 相同的token在不同位置
embeddings_same = model(same_token)

print("相同token在不同位置的嵌入向量差异:")
for i in range(3):
    print(f"位置{i}: {embeddings_same[0, i, :3].detach().numpy()}")  # 只显示前3维

# 示例5: 词汇表中的特殊token示例
print("\n--- 示例5: 特殊token示例 ---")
special_tokens = {
    "PAD": 0,    # 填充token
    "UNK": 1,    # 未知词token  
    "BOS": 2,    # 句子开始token
    "EOS": 3,    # 句子结束token
}

for name, token_id in special_tokens.items():
    token_tensor = torch.tensor([[token_id]])
    embedding = model(token_tensor)
    print(f"{name} (id={token_id}) 嵌入向量模长: {torch.norm(embedding).item():.4f}")

print(f"\n总结: 每个token都被转换为{d_model}维的密集向量表示")

# 示例6: 详细分析多维向量特性
print("\n--- 示例6: 多维向量详细分析 ---")
sample_tokens = torch.tensor([[10, 50, 100, 200, 300]])  # 5个不同的token
sample_embeddings = model(sample_tokens)

print("Token ID -> 512维向量的转换:")
for i, token_id in enumerate([10, 50, 100, 200, 300]):
    embedding_vector = sample_embeddings[0, i, :]  # 获取第i个token的嵌入向量
    
    print(f"\nToken {token_id}:")
    print(f"  向量维度: {embedding_vector.shape}")
    print(f"  向量模长: {torch.norm(embedding_vector).item():.4f}")
    print(f"  前10维: {embedding_vector[:10].detach().numpy()}")
    print(f"  最后10维: {embedding_vector[-10:].detach().numpy()}")
    print(f"  均值: {embedding_vector.mean().item():.4f}")
    print(f"  标准差: {embedding_vector.std().item():.4f}")

# 示例7: 计算token之间的相似度
print("\n--- 示例7: Token相似度计算 ---")
token1_emb = sample_embeddings[0, 0, :]  # token 10
token2_emb = sample_embeddings[0, 1, :]  # token 50
token3_emb = sample_embeddings[0, 2, :]  # token 100

# 计算余弦相似度
def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

sim_1_2 = cosine_similarity(token1_emb, token2_emb)
sim_1_3 = cosine_similarity(token1_emb, token3_emb)
sim_2_3 = cosine_similarity(token2_emb, token3_emb)

print(f"Token 10 与 Token 50 的余弦相似度: {sim_1_2.item():.4f}")
print(f"Token 10 与 Token 100 的余弦相似度: {sim_1_3.item():.4f}")
print(f"Token 50 与 Token 100 的余弦相似度: {sim_2_3.item():.4f}")

print("\n=== 完整示例总结 ===")
print(f"✅ 词汇表大小: {vocab_size} 个token")
print(f"✅ 每个token映射到: {d_model}维向量空间")
print(f"✅ 支持最大序列长度: {max_len}")
print(f"✅ 包含位置编码: 相同token在不同位置有不同表示")
print(f"✅ 向量化表示: 便于神经网络处理和计算相似度")