import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Fixed number of conv layers (hardcoded)
#
# Network Architecture Diagram:
# ┌─────────────┐
# │   Input     │  1×28×28 (grayscale image)
# │   1×28×28   │
# └──────┬──────┘
#        │
#        ▼
# ┌─────────────────────────────────┐
# │  Conv1: 1→32 ch, 3×3, pad=1    │  32×28×28
# │  ReLU                            │
# │  MaxPool2d(2,2)                  │  32×14×14
# └──────────────┬──────────────────┘
#                │
#                ▼
# ┌─────────────────────────────────┐
# │  Conv2: 32→64 ch, 3×3, pad=1   │  64×14×14
# │  ReLU                            │
# │  MaxPool2d(2,2)                  │  64×7×7
# └──────────────┬──────────────────┘
#                │
#                ▼
# ┌─────────────────────────────────┐
# │  Conv3: 64→128 ch, 3×3, pad=1  │  128×7×7
# │  ReLU                            │
# │  MaxPool2d(2,2)                  │  128×3×3
# └──────────────┬──────────────────┘
#                │
#                ▼
# ┌─────────────────────────────────┐
# │  Flatten                        │  1152 (128×3×3)
# └──────────────┬──────────────────┘
#                │
#                ▼
# ┌─────────────────────────────────┐
# │  FC1: 1152→256                  │  256
# │  ReLU                            │
# │  Dropout(0.5)                    │
# └──────────────┬──────────────────┘
#                │
#                ▼
# ┌─────────────────────────────────┐
# │  FC2: 256→10                    │  10 (num_classes)
# └─────────────────────────────────┘
#
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Manually define each conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1 channel, Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Input: 32 channels, Output: 64 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Input: 64 channels, Output: 128 channels
        
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # Calculate the size after conv layers (for 28x28 input)
        # After 3 conv+pool layers: 28 -> 14 -> 7 -> 3 (with padding)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Apply conv layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten and apply fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Method 2: Configurable number of conv layers using nn.Sequential
class ConfigurableConvNet(nn.Module):
    def __init__(self, num_conv_layers=3, input_channels=1, num_classes=10):
        super().__init__()
        
        # Build conv layers dynamically
        layers = []
        in_channels = input_channels
        out_channels = 32
        
        for i in range(num_conv_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
            out_channels *= 2  # Double the channels each layer
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size after conv layers
        self.final_channels = in_channels
        # For 28x28 input, after n pooling operations: 28 / (2^n)
        self.final_size = 28 // (2 ** num_conv_layers)
        
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.final_channels * self.final_size * self.final_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Method 3: Using ModuleList for even more flexibility
class FlexibleConvNet(nn.Module):
    def __init__(self, conv_configs, num_classes=10):
        """
        conv_configs: List of tuples (in_channels, out_channels, kernel_size, stride, padding)
        Example: [(1, 32, 3, 1, 1), (32, 64, 3, 1, 1), (64, 128, 5, 1, 2)]
        """
        super().__init__()
        
        # Create conv layers from config
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        for in_ch, out_ch, kernel, stride, padding in conv_configs:
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel, stride, padding))
            self.pool_layers.append(nn.MaxPool2d(2, 2))
        
        # Calculate final feature map size (assuming 28x28 input)
        final_channels = conv_configs[-1][1]  # Output channels of last conv layer
        final_size = 28 // (2 ** len(conv_configs))  # Size after pooling
        
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * final_size * final_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Apply each conv layer with ReLU and pooling
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = pool(F.relu(conv(x)))
        
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Method 4: Dynamic conv layers with different architectures
class AdaptiveConvNet(nn.Module):
    def __init__(self, num_layers=3, base_channels=32, growth_factor=2, 
                 input_channels=1, num_classes=10, use_batch_norm=True):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Build layers dynamically
        self.features = nn.ModuleList()
        in_ch = input_channels
        
        for i in range(num_layers):
            out_ch = base_channels * (growth_factor ** i)
            
            # Conv block
            block = []
            block.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            
            if use_batch_norm:
                block.append(nn.BatchNorm2d(out_ch))
            
            block.extend([
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            
            self.features.append(nn.Sequential(*block))
            in_ch = out_ch
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        for feature_block in self.features:
            x = feature_block(x)
        
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# Method 5: ResNet-style blocks with configurable depth
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class ConfigurableResNet(nn.Module):
    def __init__(self, num_blocks_per_layer=[2, 2, 2], channels=[32, 64, 128], 
                 input_channels=1, num_classes=10):
        super().__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        
        # Build residual layers
        self.layers = nn.ModuleList()
        in_ch = channels[0]
        
        for i, (num_blocks, out_ch) in enumerate(zip(num_blocks_per_layer, channels)):
            blocks = []
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1  # Downsample at start of each layer
                blocks.append(ResidualBlock(in_ch, out_ch, stride))
                in_ch = out_ch
            
            self.layers.append(nn.Sequential(*blocks))
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def demonstrate_different_architectures():
    """Show different ways to configure conv layers"""
    print("=" * 80)
    print("DIFFERENT WAYS TO CONFIGURE CONVOLUTIONAL LAYERS")
    print("=" * 80)
    
    # Test input (batch_size=2, channels=1, height=28, width=28)
    test_input = torch.randn(2, 1, 28, 28)
    
    print(f"Input shape: {test_input.shape}")
    
    # Method 1: Simple fixed architecture
    print("\n1. Simple ConvNet (3 fixed conv layers):")
    model1 = SimpleConvNet(num_classes=10)
    output1 = model1(test_input)
    print(f"   Output shape: {output1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Method 2: Configurable number of layers
    print("\n2. Configurable ConvNet:")
    for num_layers in [2, 4, 5]:
        model2 = ConfigurableConvNet(num_conv_layers=num_layers, num_classes=10)
        try:
            output2 = model2(test_input)
            print(f"   {num_layers} layers - Output: {output2.shape}, Params: {sum(p.numel() for p in model2.parameters()):,}")
        except Exception as e:
            print(f"   {num_layers} layers - Error: {e}")
    
    # Method 3: Flexible configuration
    print("\n3. Flexible ConvNet with custom configs:")
    configs = [
        [(1, 16, 3, 1, 1), (16, 32, 3, 1, 1)],  # 2 layers
        [(1, 32, 3, 1, 1), (32, 64, 3, 1, 1), (64, 128, 3, 1, 1)],  # 3 layers
        [(1, 64, 5, 1, 2), (64, 128, 3, 1, 1), (128, 256, 3, 1, 1), (256, 512, 3, 1, 1)]  # 4 layers
    ]
    
    for i, config in enumerate(configs):
        try:
            model3 = FlexibleConvNet(config, num_classes=10)
            output3 = model3(test_input)
            print(f"   Config {i+1} - Output: {output3.shape}, Params: {sum(p.numel() for p in model3.parameters()):,}")
        except Exception as e:
            print(f"   Config {i+1} - Error: {e}")
    
    # Method 4: Adaptive architecture
    print("\n4. Adaptive ConvNet:")
    for num_layers in [2, 3, 4]:
        model4 = AdaptiveConvNet(num_layers=num_layers, base_channels=32, num_classes=10)
        output4 = model4(test_input)
        print(f"   {num_layers} layers - Output: {output4.shape}, Params: {sum(p.numel() for p in model4.parameters()):,}")
    
    # Method 5: ResNet-style
    print("\n5. Configurable ResNet:")
    resnet_configs = [
        ([1, 1], [32, 64]),      # Small ResNet
        ([2, 2], [32, 64]),      # Medium ResNet
        ([2, 2, 2], [32, 64, 128])  # Larger ResNet
    ]
    
    for i, (blocks, channels) in enumerate(resnet_configs):
        model5 = ConfigurableResNet(num_blocks_per_layer=blocks, channels=channels, num_classes=10)
        output5 = model5(test_input)
        print(f"   ResNet-{i+1} - Output: {output5.shape}, Params: {sum(p.numel() for p in model5.parameters()):,}")

def show_conv_layer_basics():
    """Explain conv layer parameters"""
    print("\n" + "=" * 80)
    print("CONVOLUTIONAL LAYER PARAMETERS")
    print("=" * 80)
    
    print("""
Conv2d Parameters:
- in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
- out_channels: Number of output feature maps (you choose this)
- kernel_size: Size of convolution filter (e.g., 3, 5, 7)
- stride: Step size for convolution (default=1)
- padding: Zero-padding around input (default=0)

Example architectures:
- LeNet: 2 conv layers
- AlexNet: 5 conv layers  
- VGG16: 13 conv layers
- ResNet50: 49 conv layers
- You can have as many as you want!
    """)
    
    # Show effect of different parameters
    input_tensor = torch.randn(1, 1, 28, 28)
    
    print("Effect of different conv configurations on 28x28 input:")
    
    configs = [
        (1, 32, 3, 1, 1, "Basic 3x3 conv"),
        (1, 64, 5, 1, 2, "5x5 conv with padding"),
        (1, 16, 7, 1, 3, "7x7 conv with padding"),
        (1, 32, 3, 2, 1, "3x3 conv with stride=2"),
    ]
    
    for in_ch, out_ch, kernel, stride, padding, description in configs:
        conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        output = conv(input_tensor)
        print(f"  {description}: {input_tensor.shape} -> {output.shape}")

if __name__ == "__main__":
    demonstrate_different_architectures()
    show_conv_layer_basics()

