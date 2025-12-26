# Tech and Terms for distributed training.

## tech

### DiLoCo

DiLoCo（Distributed Low-Communication，分布式低通信优化）是由 DeepMind 于 2023 年提出的一种分布式优化算法，旨在解决大规模语言模型（LLM）在异构、低带宽分布式环境下的训练挑战。

### PyTorch FSDP/DDP/DP

DP: Data parallel.

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

### DeepSpeed

## projects

### DeAI

PrimeIntellect-ai/ZeroBand

### federated training
