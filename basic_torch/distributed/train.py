# train.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup(rank, world_size):
    # 初始化进程组，必须使用 gloo（CPU 环境）
    dist.init_process_group(
        backend="gloo",  # ← 关键：CPU 用 gloo
        init_method="env://",  # torchrun 会自动设置环境变量
        rank=rank,
        world_size=world_size,
    )
    # 可以不设置 device，也可以显式设为 cpu
    torch.manual_seed(42 + rank)  # 不同进程不同种子，避免完全相同


def cleanup():
    dist.destroy_process_group()


def main():
    # torchrun 会自动设置这些环境变量
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Rank {rank}, Local Rank {local_rank}, World Size {world_size}")

    setup(rank, world_size)

    # ---------------- 示例：简单线性模型 + 假数据 ----------------
    model = nn.Linear(10, 1)
    model = model.to("cpu")  # 明确放到 CPU

    # 包装成 DDP（即使是 CPU 也推荐用 DDP 来同步梯度）
    model = DDP(model, device_ids=None)  # CPU 时 device_ids 可以不填或 None

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 假数据：每个进程有自己的子集（通过 DistributedSampler 自动切分）
    data = torch.randn(1000, 10)
    labels = torch.randn(1000, 1)
    dataset = TensorDataset(data, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 训练循环
    for epoch in range(3):
        sampler.set_epoch(epoch)  # 重要：每个 epoch 打乱不同
        total_loss = 0.0
        for batch_data, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = nn.MSELoss()(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 所有进程同步 loss 打印（可选，用于观察）
        avg_loss = torch.tensor(total_loss)
        dist.all_reduce(avg_loss)
        avg_loss /= world_size

        if rank == 0:
            print(f"Epoch {epoch}, Avg Loss: {avg_loss.item():.4f}")

    if rank == 0:
        print("训练完成")

    cleanup()


if __name__ == "__main__":
    main()
