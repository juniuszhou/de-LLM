import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank} of {world_size} is running")
    dist.barrier()
    print(f"Rank {rank} is done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
