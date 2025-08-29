import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f"🔹 Forward pass started with input shape: {x.shape}")
        
        # Step 1: First linear layer
        x = self.linear1(x)
        print(f"🔹 After linear1: {x.shape}")
        
        # Step 2: Activation function
        x = self.relu(x)
        print(f"🔹 After ReLU: {x.shape}")
        
        # Step 3: Second linear layer
        x = self.linear2(x)
        print(f"🔹 After linear2 (output): {x.shape}")
        
        return x

def demonstrate_forward_behavior():
    """Show how forward function works and when it's called"""
    print("=" * 60)
    print("FORWARD FUNCTION DEMONSTRATION")
    print("=" * 60)
    
    model = SimpleNetwork()
    input_tensor = torch.randn(2, 4)  # Batch size 2, features 4
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print("\n--- Method 1: Using model(input) [RECOMMENDED] ---")
    output1 = model(input_tensor)  # This automatically calls forward()
    
    print(f"\nOutput shape: {output1.shape}")
    
    print("\n--- Method 2: Calling forward() directly [NOT RECOMMENDED] ---")
    output2 = model.forward(input_tensor)  # Direct call - avoid this
    
    print(f"\nOutput shape: {output2.shape}")
    
    print("\n--- Why model(x) is better than model.forward(x) ---")
    print("✅ model(x) triggers hooks and other PyTorch magic")
    print("❌ model.forward(x) bypasses important PyTorch functionality")

def show_forward_with_different_architectures():
    """Show forward functions for different network types"""
    print("\n" + "=" * 60)
    print("DIFFERENT FORWARD IMPLEMENTATIONS")
    print("=" * 60)
    
    # Sequential approach
    class SequentialNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 1)
            )
        
        def forward(self, x):
            # Simple: just pass through the sequential layers
            return self.network(x)
    
    # Manual step-by-step approach
    class ManualNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(4, 8)
            self.layer2 = nn.Linear(8, 4)
            self.layer3 = nn.Linear(4, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # Manual control over each step
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.layer3(x)  # No activation on output
            return x
    
    # Conditional forward (advanced)
    class ConditionalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(4, 8)
            self.layer2 = nn.Linear(8, 4)
            self.layer3 = nn.Linear(4, 1)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            
            # Conditional logic in forward pass
            if self.training:  # Only apply dropout during training
                x = self.dropout(x)
            
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    input_tensor = torch.randn(3, 4)
    
    print("\n1. Sequential Network:")
    seq_net = SequentialNet()
    output = seq_net(input_tensor)
    print(f"   Output shape: {output.shape}")
    
    print("\n2. Manual Network:")
    manual_net = ManualNet()
    output = manual_net(input_tensor)
    print(f"   Output shape: {output.shape}")
    
    print("\n3. Conditional Network (Training mode):")
    cond_net = ConditionalNet()
    cond_net.train()  # Set to training mode
    output = cond_net(input_tensor)
    print(f"   Output shape: {output.shape}")
    
    print("\n4. Conditional Network (Evaluation mode):")
    cond_net.eval()  # Set to evaluation mode
    output = cond_net(input_tensor)
    print(f"   Output shape: {output.shape}")

def show_gradient_flow():
    """Demonstrate how forward enables gradient computation"""
    print("\n" + "=" * 60)
    print("FORWARD FUNCTION AND GRADIENTS")
    print("=" * 60)
    
    class GradientDemo(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
        
        def forward(self, x):
            # The forward function defines the computation graph
            # PyTorch tracks operations for backward pass
            return self.linear(x)
    
    model = GradientDemo()
    
    # Input with gradient tracking
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[3.0]])
    
    print(f"Input: {x}")
    print(f"Target: {target}")
    
    # Forward pass
    output = model(x)  # This calls forward() and builds computation graph
    print(f"Model output: {output}")
    
    # Compute loss
    loss = nn.MSELoss()(output, target)
    print(f"Loss: {loss}")
    
    # Backward pass - this uses the computation graph built in forward()
    loss.backward()
    
    print("\nGradients computed:")
    print(f"Input gradient: {x.grad}")
    for name, param in model.named_parameters():
        print(f"{name} gradient: {param.grad}")

def compare_with_functions():
    """Compare neural network forward with regular functions"""
    print("\n" + "=" * 60)
    print("FORWARD vs REGULAR FUNCTION")
    print("=" * 60)
    
    # Regular Python function
    def regular_function(x):
        return x * 2 + 1
    
    # Neural network "function"
    class NetworkFunction(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.bias = nn.Parameter(torch.tensor(1.0))
        
        def forward(self, x):
            return x * self.weight + self.bias
    
    x = torch.tensor(5.0)
    
    # Regular function
    result1 = regular_function(x)
    print(f"Regular function result: {result1}")
    
    # Neural network
    model = NetworkFunction()
    result2 = model(x)  # Calls forward()
    print(f"Neural network result: {result2}")
    
    print("\nKey differences:")
    print("✅ Neural network: Can learn parameters via gradients")
    print("✅ Neural network: Integrates with PyTorch ecosystem")
    print("✅ Neural network: Can be saved/loaded")
    print("❌ Regular function: Fixed behavior, no learning")

if __name__ == "__main__":
    demonstrate_forward_behavior()
    show_forward_with_different_architectures()
    show_gradient_flow()
    compare_with_functions()

