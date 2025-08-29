import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        # Initialize with known weights for demonstration
        with torch.no_grad():
            self.linear.weight.data = torch.tensor([[1.0, 2.0]])
            self.linear.bias.data = torch.tensor([0.0])
    
    def forward(self, x):
        return self.linear(x)

def demonstrate_gradient_accumulation():
    """Show what happens WITHOUT zero_grad()"""
    print("=" * 60)
    print("GRADIENT ACCUMULATION WITHOUT zero_grad()")
    print("=" * 60)
    
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # Same input and target for multiple iterations
    x = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[0.0]])
    
    print("Initial weight:", model.linear.weight.data)
    print("Initial bias:", model.linear.bias.data)
    
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Forward pass
        output = model(x)
        loss = criterion(output, target)
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass (accumulates gradients)
        loss.backward()
        
        # Check gradients BEFORE zero_grad()
        print(f"Weight gradient: {model.linear.weight.grad}")
        print(f"Bias gradient: {model.linear.bias.grad}")
        
        # Optimizer step
        optimizer.step()
        
        print(f"Updated weight: {model.linear.weight.data}")
        print(f"Updated bias: {model.linear.bias.data}")
        
        # ❌ NO zero_grad() - gradients keep accumulating!

def demonstrate_with_zero_grad():
    """Show what happens WITH zero_grad()"""
    print("\n" + "=" * 60)
    print("PROPER TRAINING WITH zero_grad()")
    print("=" * 60)
    
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    x = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[0.0]])
    
    print("Initial weight:", model.linear.weight.data)
    print("Initial bias:", model.linear.bias.data)
    
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # ✅ Clear gradients from previous step
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)
        loss = criterion(output, target)
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients AFTER zero_grad() and backward()
        print(f"Weight gradient: {model.linear.weight.grad}")
        print(f"Bias gradient: {model.linear.bias.grad}")
        
        # Optimizer step
        optimizer.step()
        
        print(f"Updated weight: {model.linear.weight.data}")
        print(f"Updated bias: {model.linear.bias.data}")

def show_gradient_accumulation_example():
    """Show when gradient accumulation might be useful"""
    print("\n" + "=" * 60)
    print("INTENTIONAL GRADIENT ACCUMULATION")
    print("=" * 60)
    
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # Simulate large batch by accumulating gradients from multiple mini-batches
    mini_batches = [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[2.0, 1.0]]), torch.tensor([[1.0]])),
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0]]))
    ]
    
    print("Accumulating gradients over 3 mini-batches:")
    
    # Clear gradients once at the beginning
    optimizer.zero_grad()
    
    total_loss = 0
    for i, (x, target) in enumerate(mini_batches):
        print(f"\nMini-batch {i + 1}:")
        
        # Forward pass
        output = model(x)
        loss = criterion(output, target)
        total_loss += loss.item()
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass (accumulates gradients)
        loss.backward()
        
        print(f"  Weight gradient: {model.linear.weight.grad}")
        print(f"  Bias gradient: {model.linear.bias.grad}")
    
    print(f"\nTotal accumulated loss: {total_loss:.4f}")
    print(f"Final accumulated gradients:")
    print(f"  Weight: {model.linear.weight.grad}")
    print(f"  Bias: {model.linear.bias.grad}")
    
    # Single optimizer step on accumulated gradients
    optimizer.step()
    print(f"Updated weight: {model.linear.weight.data}")
    print(f"Updated bias: {model.linear.bias.data}")

def compare_set_to_none():
    """Compare zero_grad() vs zero_grad(set_to_none=False)"""
    print("\n" + "=" * 60)
    print("zero_grad() vs zero_grad(set_to_none=False)")
    print("=" * 60)
    
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    x = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[0.0]])
    
    # First iteration
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    
    print("After backward pass:")
    print(f"Weight grad: {model.linear.weight.grad}")
    print(f"Bias grad: {model.linear.bias.grad}")
    
    # Method 1: set_to_none=True (default)
    optimizer.zero_grad(set_to_none=True)
    print("\nAfter zero_grad(set_to_none=True):")
    print(f"Weight grad: {model.linear.weight.grad}")
    print(f"Bias grad: {model.linear.bias.grad}")
    
    # Another backward pass
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    
    # Method 2: set_to_none=False
    optimizer.zero_grad(set_to_none=False)
    print("\nAfter zero_grad(set_to_none=False):")
    print(f"Weight grad: {model.linear.weight.grad}")
    print(f"Bias grad: {model.linear.bias.grad}")

def show_training_loop_pattern():
    """Show the standard training loop pattern"""
    print("\n" + "=" * 60)
    print("STANDARD TRAINING LOOP PATTERN")
    print("=" * 60)
    
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # Training data
    train_data = [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[2.0, 1.0]]), torch.tensor([[1.0]])),
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0]])),
    ]
    
    print("Standard PyTorch training loop:")
    print("""
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()    # 1. Clear gradients
            output = model(input)    # 2. Forward pass
            loss = criterion(output, target)  # 3. Compute loss
            loss.backward()          # 4. Backward pass
            optimizer.step()         # 5. Update parameters
    """)
    
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}:")
        epoch_loss = 0
        
        for i, (x, target) in enumerate(train_data):
            # 1. Clear gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            output = model(x)
            
            # 3. Compute loss
            loss = criterion(output, target)
            epoch_loss += loss.item()
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Update parameters
            optimizer.step()
            
            print(f"  Batch {i+1}: Loss = {loss.item():.4f}")
        
        print(f"  Average loss: {epoch_loss / len(train_data):.4f}")

if __name__ == "__main__":
    demonstrate_gradient_accumulation()
    demonstrate_with_zero_grad()
    show_gradient_accumulation_example()
    compare_set_to_none()
    show_training_loop_pattern()

