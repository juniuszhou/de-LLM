# Tech and Terms for distributed training.

## tech

### DiLoCo

DiLoCo（Distributed Low-Communication，分布式低通信优化）是由 DeepMind 于 2023 年提出的一种分布式优化算法，旨在解决大规模语言模型（LLM）在异构、低带宽分布式环境下的训练挑战。

### PyTorch FSDP/DDP/DP

DP: Data parallel. 数据并行

DPP: Distributed Data Parallel
a good mapping rule is: one process to one GPU
rank: your gpu ID in whole world according to world_size
world_size: total number of processes
local size: processes per node

FSDP:
Fully Sharded Data Parallel (FSDP) in PyTorch
Fully Sharded Data Parallel (FSDP) is a distributed training wrapper in PyTorch designed to shard (split) a model's parameters, gradients, and optimizer states across multiple GPUs or processes. This allows training very large models on hardware with limited memory per device, inspired by the ZeRO-3 technique from DeepSpeed. It's particularly useful for scaling models like large language models (LLMs) beyond what standard data parallelism can handle.

FSDP reduces the memory footprint per GPU by avoiding full model replication. Instead of duplicating the entire model on each device (as in traditional parallelism), it distributes shards dynamically during training.

---

| Feature           | DP                           | DDP                               | FSDP                              |
| ----------------- | ---------------------------- | --------------------------------- | --------------------------------- |
| Model replication | Full model on each GPU       | Full model on each GPU            | Sharded across GPUs               |
| Memory efficiency | Low (full model copy)        | Low (full model copy)             | High (model sharding)             |
| Communication     | All-reduce gradients         | All-reduce gradients              | All-gather/Reduce-scatter         |
| Scalability       | Limited by single GPU memory | Limited by single GPU memory      | Scales to very large models       |
| Setup complexity  | Simple                       | Moderate (requires process group) | Moderate (requires process group) |
| Use case          | Small models, single node    | Medium models, multi-GPU          | Large models, limited GPU memory  |

---

#### Model parallel

模型的并行和数据并行相对应，把训练任务分割。

pipeline parallet:
流水线并行，把每个层放到不同的 GPU 中训练，让 GPU 像流水线一样计算

Tensor parallel
把不同的 Tensor 放到不同的 GPU 中训练，像 transformer 这样的模型，每个头是可以单独并行计算的。

## 核心概念

数据并行 (DP)：每个设备持有完整模型副本，仅将输入数据（batch）分片。每个设备独立计算前向/反向传播，然后同步梯度（e.g., AllReduce）。
DP 都是通过不同的程序都独立的采样数据来达到 DP 的效果。

张量并行 (TP)：将模型参数（张量，如权重矩阵）分片到多个设备上，每个设备计算部分操作（e.g., 矩阵乘法的部分行/列）。需要频繁通信交换中间结果。

现代框架支持 3D 并行（DP + TP + Pipeline
否则 TP 跨节点会变慢（推荐 TP 在节点内）

### DeepSpeed

## projects

### DeAI

PrimeIntellect-ai/ZeroBand

### federated training

### embedding 的 TP 实现和机制

在 embedding 层，它只是做一个查找的操作，根据 token id 找到多维向量的表示。它的 TP 配置如下。

"tok_embeddings": RowwiseParallel( # each GPU has a copy of the token embeddings, so we use Replicate()
input_layouts=Replicate(), # output sharded according to the tp_size，Shard(0) 是 batch 的维度
output_layouts=Shard(1),
),

RowwiseParallel 意味着每个 GPU 都只有一部分 token 的维度信息，因为它是按照 vocab 来 split 的。
那么它在 embedding 之后如何找到缺失的信息呢，它并不是把所有的结果都发送给其他 GPU。
而是通过 AllGather 的方式，只向其他 GPU 查询丢失的信息，这样通信量就大大减少了。

input 使用 Replicate，因为所有的输入都需要分到所有 GPU 去做查询。
output 使用 Shard(1) 使用多维向量来划分 TP。可以和接下来的 encoder 层衔接。

##

SequenceParallel 的核心是序列级并行（sequence-level parallelism），它不分片模型参数，而是聚焦于输入数据的分片，以节省激活内存。
Transformer 模型中，激活张量（如 hidden_states）形状通常为 [batch_size, seq_len, hidden_dim]
那么它 shard 的维度就是第二个，对 token 序列进行分片，然后分别计算。 它没有对模型参数进行分片。
数据结果也不需要做 AllGather 或者 AllReduce，结果都是独立的。

不像 Rowwise 线性层的 AllReduce

#### ColwiseParallel 一般用在 Linear 模型

ColwiseParallel 把 weight（形状 [in_features, out_features]）沿 out_features（dim=1，列维度） 分片
它默认的输入是 replica，复制整个数据。输出是沿着 Shard（1）分片

forward 没有 reduce，它的输出是分片的。
backward 有梯度聚合。
