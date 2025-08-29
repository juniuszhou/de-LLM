import torch
import torch.nn as nn

# ✅ Correct way - inherits from nn.Module
class GoodNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # CRITICAL LINE
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# ❌ Wrong way - no nn.Module inheritance
class BadNetwork:
    def __init__(self):
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def demonstrate_difference():
    print("=== GOOD NETWORK (with nn.Module) ===")
    good_model = GoodNetwork()
    
    print(f"Parameters count: {sum(p.numel() for p in good_model.parameters())}")
    print(f"Can list parameters: {len(list(good_model.parameters()))} tensors")
    print(f"Has named_parameters: {len(list(good_model.named_parameters()))} items")
    print(f"Can move to device: {hasattr(good_model, 'to')}")
    print(f"Has training mode: {hasattr(good_model, 'train')}")
    
    print("\n=== BAD NETWORK (without nn.Module) ===")
    bad_model = BadNetwork()
    
    try:
        params = list(bad_model.parameters())
        print(f"Parameters: {len(params)}")
    except AttributeError as e:
        print(f"❌ Error: {e}")
    
    try:
        bad_model.to('cpu')
        print("✅ Can move to device")
    except AttributeError as e:
        print(f"❌ Error: {e}")
    
    try:
        bad_model.train()
        print("✅ Has training mode")
    except AttributeError as e:
        print(f"❌ Error: {e}")

def show_parameter_details():
    print("\n=== PARAMETER TRACKING DETAILS ===")
    model = GoodNetwork()
    
    print("Named parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} | requires_grad: {param.requires_grad}")
    
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Show how gradients work
    x = torch.randn(1, 4)
    y_true = torch.randn(1, 1)
    
    # Forward pass
    y_pred = model(x)
    loss = nn.MSELoss()(y_pred, y_true)
    
    # Backward pass
    loss.backward()
    
    print("\nGradients after backward pass:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad shape {param.grad.shape}, grad norm: {param.grad.norm():.4f}")

if __name__ == "__main__":
    demonstrate_difference()
    show_parameter_details()

