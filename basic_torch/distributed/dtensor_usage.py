"""
DTensor Usage Examples and Basic Operations

DTensor (Distributed Tensor) is PyTorch's API for distributed tensor operations.
It allows tensors to be sharded across multiple devices/processes.

Note: DTensor requires distributed initialization (torch.distributed.init_process_group)
"""

from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.distributed.device_mesh import init_device_mesh
import torch
import os

# ============================================================================
# Note: get_event_loop vs get_running_loop (for asyncio context)
# ============================================================================
"""
get_event_loop() vs get_running_loop():

- get_event_loop(): Gets or creates an event loop (may create new one)
- get_running_loop(): Gets the currently running event loop (raises error if none)

Use get_running_loop() when you're sure a loop is running (inside async function).
Use get_event_loop() when you need to get/create a loop (but deprecated in Python 3.10+).
"""


# ============================================================================
# Example 1: Basic DTensor Creation
# ============================================================================


def example_basic_dtensor():
    """Basic DTensor creation with DeviceMesh"""
    print("=" * 70)
    print("Example 1: Basic DTensor Creation")
    print("=" * 70)

    # Initialize distributed (required for DTensor)
    # In real usage, this would be done before creating DTensors
    if not torch.distributed.is_initialized():
        # For single process demo, we'll skip actual initialization
        print("Note: Distributed not initialized - showing conceptual usage")
        return

    # Create device mesh (2D mesh for 4 devices)
    device_mesh = init_device_mesh("cpu", mesh_shape=(2, 1))

    # Create a regular tensor
    local_tensor = torch.randint(0, 10, (4, 4))

    # Create DTensor with Shard placement (shard along dimension 0)
    dtensor = DTensor.from_local(
        local_tensor,
        device_mesh=device_mesh,
        placements=[Shard(0)],  # Shard along first dimension
    )
    print(dtensor)
    dtensor = dtensor + dtensor
    print(dtensor)

    print(f"Local tensor shape: {local_tensor.shape}")
    print(f"DTensor shape: {dtensor.shape}")
    print(f"DTensor device mesh: {dtensor.device_mesh}")
    print(f"DTensor placements: {dtensor.placements}")


# ============================================================================
# Example 2: Different Sharding Strategies
# ============================================================================


def example_sharding_strategies():
    """Demonstrate different sharding strategies"""
    print("\n" + "=" * 70)
    print("Example 2: Sharding Strategies")
    print("=" * 70)

    print(
        """
    Sharding Strategies:
    
    1. Shard(dim): Shard tensor along specified dimension
       - Shard(0): Shard along rows (each device gets some rows)
       - Shard(1): Shard along columns (each device gets some columns)
    
    2. Replicate(): Replicate tensor on all devices
       - Full copy on each device
    
    3. Partial(): Partial replication (for reduction operations)
    """
    )


# ============================================================================
# Example 3: DTensor Operations
# ============================================================================


def example_dtensor_operations():
    """Basic operations with DTensor"""
    print("\n" + "=" * 70)
    print("Example 3: DTensor Operations")
    print("=" * 70)

    print(
        """
    Basic Operations (similar to regular tensors):
    
    1. Arithmetic Operations:
       dtensor1 + dtensor2
       dtensor * scalar
       dtensor @ other_dtensor  # Matrix multiplication
    
    2. Element-wise Operations:
       torch.relu(dtensor)
       torch.sin(dtensor)
       dtensor.sum()
       dtensor.mean()
    
    3. Redistribute:
       dtensor.redistribute(placements=[Shard(1)])  # Change sharding
    
    4. Convert to Local:
       local_tensor = dtensor.to_local()  # Get local shard
    """
    )


# ============================================================================
# Example 4: Complete DTensor Workflow
# ============================================================================


def example_complete_workflow():
    """Complete workflow example"""
    print("\n" + "=" * 70)
    print("Example 4: Complete DTensor Workflow")
    print("=" * 70)

    code_example = """
    # Step 1: Initialize distributed
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    
    # Step 2: Create device mesh
    device_mesh = init_device_mesh("cuda", mesh_shape=(2,))
    
    # Step 3: Create local tensor
    local_tensor = torch.randn(8, 4, device="cuda")
    
    # Step 4: Create DTensor (sharded along dim 0)
    dtensor = DTensor.from_local(
        local_tensor,
        device_mesh=device_mesh,
        placements=[Shard(0)]
    )
    
    # Step 5: Perform operations
    result = torch.relu(dtensor)
    result = result @ torch.randn(4, 8)  # Matrix multiplication
    
    # Step 6: Redistribute if needed
    replicated = result.redistribute(placements=[Replicate()])
    
    # Step 7: Get local shard
    local_result = replicated.to_local()
    """

    print(code_example)


# ============================================================================
# Example 5: DTensor with Neural Networks
# ============================================================================


def example_with_nn():
    """Using DTensor with neural network layers"""
    print("\n" + "=" * 70)
    print("Example 5: DTensor with Neural Networks")
    print("=" * 70)

    code_example = """
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module
    )
    import torch.nn as nn
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create device mesh
    device_mesh = init_device_mesh("cuda", mesh_shape=(2,))
    
    # Parallelize model
    parallel_model = parallelize_module(
        model,
        device_mesh,
        parallelize_plan={
            "0": ColwiseParallel(),  # Column-wise parallel
            "2": RowwiseParallel(),  # Row-wise parallel
        }
    )
    
    # Create DTensor input
    input_dtensor = DTensor.from_local(
        torch.randn(4, 10),
        device_mesh=device_mesh,
        placements=[Shard(0)]
    )
    
    # Forward pass
    output = parallel_model(input_dtensor)
    """

    print(code_example)


# ============================================================================
# Example 6: Comparison: Regular Tensor vs DTensor
# ============================================================================


def example_comparison():
    """Compare regular tensor operations with DTensor"""
    print("\n" + "=" * 70)
    print("Example 6: Regular Tensor vs DTensor")
    print("=" * 70)

    print(
        """
    Regular Tensor:
    ┌─────────────────┐
    │  Full tensor    │  (on single device)
    │  [8, 4]         │
    └─────────────────┘
    
    DTensor (Shard(0)):
    ┌─────────┬─────────┐
    │ Shard 0 │ Shard 1 │  (distributed across devices)
    │ [4, 4]  │ [4, 4]  │
    └─────────┴─────────┘
      Device0  Device1
    
    Operations:
    - Regular: All computation on one device
    - DTensor: Computation distributed, results aggregated automatically
    """
    )


# ============================================================================
# Example 7: Common Patterns
# ============================================================================


def example_common_patterns():
    """Common DTensor usage patterns"""
    print("\n" + "=" * 70)
    print("Example 7: Common Patterns")
    print("=" * 70)

    patterns = """
    Pattern 1: Data Parallel (batch dimension sharded)
    ----------------------------------------------------
    dtensor = DTensor.from_local(
        batch_data,  # [batch_size, features]
        device_mesh,
        placements=[Shard(0)]  # Each device gets part of batch
    )
    
    Pattern 2: Tensor Parallel (feature dimension sharded)
    --------------------------------------------------------
    dtensor = DTensor.from_local(
        weight_matrix,  # [out_features, in_features]
        device_mesh,
        placements=[Shard(1)]  # Each device gets part of input features
    )
    
    Pattern 3: Pipeline Parallel (replicated)
    -------------------------------------------
    dtensor = DTensor.from_local(
        layer_output,
        device_mesh,
        placements=[Replicate()]  # Full copy on each device
    )
    
    Pattern 4: 2D Parallelism
    --------------------------
    device_mesh = init_device_mesh("cuda", mesh_shape=(2, 2))
    dtensor = DTensor.from_local(
        tensor,
        device_mesh,
        placements=[Shard(0), Shard(1)]  # 2D sharding
    )
    """

    print(patterns)


# ============================================================================
# Main Function
# torchrun --nnodes=1 --nproc_per_node=${1:-1} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${1:-dtensor_usage.py}
# ============================================================================


def main():
    """Run all examples"""
    print("DTensor Usage Examples")
    print("=" * 70)
    print("\nNote: These examples show conceptual usage.")
    print("For actual execution, you need:")
    print("  1. Multiple GPUs/devices")
    print("  2. Distributed initialization (torch.distributed.init_process_group)")
    print("  3. Run with torchrun or similar\n")

    torch.distributed.init_process_group(backend="gloo")

    example_basic_dtensor()
    # example_sharding_strategies()
    # example_dtensor_operations()
    # example_complete_workflow()
    # example_with_nn()
    # example_comparison()
    # example_common_patterns()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
