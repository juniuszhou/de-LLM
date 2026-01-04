import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, Replicate
import os


def mesh_1d():
    # 步骤 1: 初始化分布式（在每个进程中运行）
    local_rank = int(os.environ["LOCAL_RANK"])  # 通过 torchrun 设置
    dist.init_process_group(backend="nccl", init_method="env://")

    # 步骤 2: 创建 1D 网格（假设 2 个 GPU）
    world_size = dist.get_world_size()  # 总进程数
    mesh = DeviceMesh("cuda", torch.arange(world_size))  # tensor([0, 1])

    # 步骤 3: 创建本地张量（每个 rank 持有部分数据）
    local_tensor = torch.randn(4, 8)  # 示例形状

    # 步骤 4: 使用 Shard 在维度 0 上分片创建 DTensor
    dtensor = DTensor.from_local(
        local_tensor, device_mesh=mesh, placements=[Shard(0)]  # 在网格维度 0 上分片
    )

    # 使用 dtensor 计算（自动分布式）
    result = dtensor * 2
    print(
        f"Local result shape: {result.local_tensor().shape}"
    )  # 每个 rank 打印本地部分


def mesh_2d():
    dist.init_process_group(backend="nccl")

    # 创建 2D 网格：torch.tensor([[0,1], [2,3]]) 表示 2 行（数据并行）x 2 列（张量并行）

    mesh_2d = torch.tensor([[0, 1], [2, 3]])
    device_mesh = DeviceMesh("cuda", mesh_2d)

    # 示例：输入分片（数据并行），权重复制（张量并行）
    input_dtensor = DTensor.from_local(
        torch.randn(4, 8),
        device_mesh=device_mesh,
        placements=[Replicate()],  # 复制到所有设备
    )

    weight_dtensor = DTensor.from_local(
        torch.randn(8, 16),
        device_mesh=device_mesh,
        placements=[Shard(0)],  # 在列维度（1）上分片权重
    )

    # 矩阵乘法（自动处理通信）
    output = torch.mm(input_dtensor, weight_dtensor)
    print(output.shape)  # (4, 16)，分布式执行


def main():
    # 在 PyTorch 2.4+ 的 Fully Sharded Data Parallel (FSDP2) 中，可使用预定义全局网格：
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from torch.distributed.device_mesh import init_device_mesh

    # 自动初始化全局网格（基于环境变量）
    mesh = init_device_mesh("cuda")  # 自动检测 world_size

    # 在模型中使用
    model = YourModel()
    fsdp_model = FSDP(model, device_mesh=mesh, auto_wrap_policy=...)  # 分片策略
