import torch
import torch.nn as nn

# 在神经网络（如 Transformer）中，Embedding（嵌入层）是将离散输入（如 token ID）映射为连续向量的可学习组件。
# 其训练过程本质上是通过反向传播（Backpropagation）更新嵌入矩阵，使向量捕捉输入的语义或特征表示。
# Embedding 不是孤立训练的，而是嵌入整个模型的端到端训练中，与其他层（如注意力、FFN）共同优化。

# 创建 Embedding 层
vocab_size = 100
embedding_dim = 100
embedding = nn.Embedding(vocab_size, embedding_dim)

# 输入：索引张量
input_indices = torch.tensor([0, 2, 4])  # 形状: (3,)

print(input_indices.shape)  # ([3,])
# 前向传播
output = embedding(input_indices)
print(output)
print(output.shape)  # ([3, 100])

# vocab_size=10000: 参数=7,680,000, 内存≈29.2 MB
# vocab_size=30000: 参数=23,040,000, 内存≈87.7 MB
# vocab_size=100000: 参数=76,800,000, 内存≈292.3 MB

# vocab_size 规模,参数/内存,训练效率,泛化能力,适用场景
# 小 (5k-20k),低,高（密集更新）,中（OOV 多）,小数据集、特定领域
# 中 (20k-50k),中,平衡,高（子词共享）,通用 NLP（如 BERT）
# 大 (50k+),高,低（稀疏）,中高（覆盖广）,多语言、大语料
