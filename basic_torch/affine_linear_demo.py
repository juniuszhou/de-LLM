import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def demonstrate_affine_linear_transformation():
    """Demonstrate the affine linear transformation mathematically"""
    print("=" * 80)
    print("AFFINE LINEAR TRANSFORMATION: Y = XW + b")
    print("=" * 80)
    
    # Define dimensions
    batch_size = 3
    input_features = 4
    output_features = 3
    
    print(f"Batch size: {batch_size}")
    print(f"Input features: {input_features}")
    print(f"Output features: {output_features}")
    
    # Create example data
    X = torch.randn(batch_size, input_features)
    print(f"\nInput X shape: {X.shape}")
    print("Input X:")
    print(X)
    
    # Create linear layer
    linear = nn.Linear(input_features, output_features)
    
    # Get weight and bias
    W = linear.weight.data  # Shape: [output_features, input_features]
    b = linear.bias.data    # Shape: [output_features]
    
    print(f"\nWeight W shape: {W.shape}")
    print("Weight W:")
    print(W)
    
    print(f"\nBias b shape: {b.shape}")
    print("Bias b:")
    print(b)
    
    # Manual computation: Y = XW^T + b
    # Note: PyTorch Linear layer uses W^T (transposed weight matrix)
    Y_manual = torch.matmul(X, W.T) + b
    
    # Using PyTorch Linear layer
    Y_pytorch = linear(X)
    
    print(f"\nOutput Y shape: {Y_pytorch.shape}")
    print("Output Y (PyTorch):")
    print(Y_pytorch)
    
    print("\nOutput Y (Manual calculation):")
    print(Y_manual)
    
    print(f"\nAre they equal? {torch.allclose(Y_pytorch, Y_manual)}")
    
    # Show step-by-step calculation for first sample
    print("\n" + "="*50)
    print("STEP-BY-STEP CALCULATION FOR FIRST SAMPLE")
    print("="*50)
    
    x_sample = X[0]  # First sample
    print(f"First input sample: {x_sample}")
    
    for i in range(output_features):
        w_row = W[i]  # i-th row of weight matrix
        b_i = b[i]    # i-th bias
        
        # Dot product + bias
        y_i = torch.dot(x_sample, w_row) + b_i
        
        print(f"\nOutput {i+1}:")
        print(f"  y_{i+1} = x·w_{i+1} + b_{i+1}")
        print(f"  y_{i+1} = {x_sample.tolist()} · {w_row.tolist()} + {b_i.item():.4f}")
        print(f"  y_{i+1} = {y_i.item():.4f}")

def show_linear_layer_internals():
    """Show what happens inside nn.Linear"""
    print("\n" + "=" * 80)
    print("INSIDE nn.Linear LAYER")
    print("=" * 80)
    
    # Create a simple linear layer
    linear = nn.Linear(in_features=4, out_features=2)
    
    print("Linear layer parameters:")
    for name, param in linear.named_parameters():
        print(f"  {name}: shape {param.shape}")
        print(f"    {param}")
    
    # Input
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(f"\nInput: {x}")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = linear(x)
    print(f"\nOutput: {output}")
    print(f"Output shape: {output.shape}")
    
    # Manual calculation
    print("\nManual calculation:")
    weight = linear.weight
    bias = linear.bias
    
    print(f"Weight matrix:\n{weight}")
    print(f"Bias vector: {bias}")
    
    # PyTorch uses: output = input @ weight.T + bias
    manual_output = x @ weight.T + bias
    print(f"Manual result: {manual_output}")
    print(f"Match PyTorch? {torch.allclose(output, manual_output)}")

def visualize_transformation_2d():
    """Visualize 2D to 2D affine transformation"""
    print("\n" + "=" * 80)
    print("2D AFFINE TRANSFORMATION VISUALIZATION")
    print("=" * 80)
    
    # Create a simple 2D to 2D transformation
    linear = nn.Linear(2, 2)
    
    # Set specific weights for visualization
    with torch.no_grad():
        linear.weight.data = torch.tensor([[1.5, 0.5], 
                                         [-0.3, 1.2]])
        linear.bias.data = torch.tensor([1.0, -0.5])
    
    print("Transformation matrix:")
    print(linear.weight.data)
    print("Bias vector:")
    print(linear.bias.data)
    
    # Create grid of input points
    x_range = torch.linspace(-2, 2, 20)
    y_range = torch.linspace(-2, 2, 20)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
    
    # Flatten and stack to create input points
    input_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Apply transformation
    output_points = linear(input_points)
    
    # Reshape for plotting
    x_out = output_points[:, 0].reshape(20, 20)
    y_out = output_points[:, 1].reshape(20, 20)
    
    print(f"Transformed {input_points.shape[0]} points")
    print("Original points range:", 
          f"x: [{input_points[:, 0].min():.2f}, {input_points[:, 0].max():.2f}]",
          f"y: [{input_points[:, 1].min():.2f}, {input_points[:, 1].max():.2f}]")
    print("Transformed points range:",
          f"x: [{output_points[:, 0].min():.2f}, {output_points[:, 0].max():.2f}]",
          f"y: [{output_points[:, 1].min():.2f}, {output_points[:, 1].max():.2f}]")

def compare_linear_vs_nonlinear():
    """Compare linear transformation with nonlinear activation"""
    print("\n" + "=" * 80)
    print("LINEAR vs NONLINEAR TRANSFORMATIONS")
    print("=" * 80)
    
    # Create input
    x = torch.linspace(-3, 3, 100).unsqueeze(1)
    
    # Linear transformation only
    linear_only = nn.Linear(1, 1)
    with torch.no_grad():
        linear_only.weight.data = torch.tensor([[2.0]])
        linear_only.bias.data = torch.tensor([1.0])
    
    y_linear = linear_only(x)
    
    # Linear + ReLU
    linear_relu = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU()
    )
    with torch.no_grad():
        linear_relu[0].weight.data = torch.tensor([[2.0]])
        linear_relu[0].bias.data = torch.tensor([1.0])
    
    y_relu = linear_relu(x)
    
    # Linear + Sigmoid
    linear_sigmoid = nn.Sequential(
        nn.Linear(1, 1),
        nn.Sigmoid()
    )
    with torch.no_grad():
        linear_sigmoid[0].weight.data = torch.tensor([[2.0]])
        linear_sigmoid[0].bias.data = torch.tensor([1.0])
    
    y_sigmoid = linear_sigmoid(x)
    
    print("Linear transformation: y = 2x + 1")
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Linear output range: [{y_linear.min():.2f}, {y_linear.max():.2f}]")
    print(f"ReLU output range: [{y_relu.min():.2f}, {y_relu.max():.2f}]")
    print(f"Sigmoid output range: [{y_sigmoid.min():.2f}, {y_sigmoid.max():.2f}]")

def show_batch_processing():
    """Show how linear layers handle batches"""
    print("\n" + "=" * 80)
    print("BATCH PROCESSING IN LINEAR LAYERS")
    print("=" * 80)
    
    linear = nn.Linear(3, 2)
    
    # Single sample
    single_sample = torch.randn(3)
    single_output = linear(single_sample)
    
    print(f"Single sample input shape: {single_sample.shape}")
    print(f"Single sample output shape: {single_output.shape}")
    
    # Batch of samples
    batch_samples = torch.randn(5, 3)  # 5 samples, 3 features each
    batch_output = linear(batch_samples)
    
    print(f"Batch input shape: {batch_samples.shape}")
    print(f"Batch output shape: {batch_output.shape}")
    
    # Show that each sample is processed independently
    print("\nVerifying batch processing:")
    for i in range(batch_samples.shape[0]):
        individual_output = linear(batch_samples[i])
        batch_row_output = batch_output[i]
        match = torch.allclose(individual_output, batch_row_output)
        print(f"Sample {i}: Individual vs Batch processing match? {match}")

if __name__ == "__main__":
    demonstrate_affine_linear_transformation()
    show_linear_layer_internals()
    visualize_transformation_2d()
    compare_linear_vs_nonlinear()
    show_batch_processing()
