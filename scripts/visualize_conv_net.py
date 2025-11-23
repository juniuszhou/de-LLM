"""
Visualization script for SimpleConvNet architecture
Generates a diagram showing the network structure and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualize_simple_conv_net():
    """Create a visual diagram of SimpleConvNet architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors for different layer types
    colors = {
        'input': '#E8F4F8',
        'conv': '#FFE5B4',
        'pool': '#C8E6C9',
        'fc': '#E1BEE7',
        'output': '#FFCDD2',
        'activation': '#FFF9C4'
    }
    
    # Layer definitions with positions and sizes
    layers = [
        # (x, y, width, height, label, color, details)
        (0.5, 5, 1.2, 1.5, 'Input\n1×28×28', colors['input'], ''),
        (2.2, 5, 1.2, 1.5, 'Conv1\n1→32 ch\n3×3, pad=1', colors['conv'], ''),
        (2.2, 3, 1.2, 0.8, 'ReLU', colors['activation'], ''),
        (2.2, 1.8, 1.2, 0.8, 'MaxPool\n2×2', colors['pool'], ''),
        (4.0, 5, 1.2, 1.5, 'Conv2\n32→64 ch\n3×3, pad=1', colors['conv'], ''),
        (4.0, 3, 1.2, 0.8, 'ReLU', colors['activation'], ''),
        (4.0, 1.8, 1.2, 0.8, 'MaxPool\n2×2', colors['pool'], ''),
        (5.8, 5, 1.2, 1.5, 'Conv3\n64→128 ch\n3×3, pad=1', colors['conv'], ''),
        (5.8, 3, 1.2, 0.8, 'ReLU', colors['activation'], ''),
        (5.8, 1.8, 1.2, 0.8, 'MaxPool\n2×2', colors['pool'], ''),
        (7.6, 5, 1.2, 1.5, 'Flatten\n1152', colors['fc'], ''),
        (7.6, 3, 1.2, 1.5, 'FC1\n1152→256', colors['fc'], ''),
        (7.6, 1, 1.2, 0.8, 'ReLU', colors['activation'], ''),
        (7.6, 0.1, 1.2, 0.6, 'Dropout\n0.5', colors['fc'], ''),
        (9.2, 3, 1.2, 1.5, 'FC2\n256→10', colors['fc'], ''),
        (9.2, 1.5, 1.2, 0.8, 'Output\n10 classes', colors['output'], ''),
    ]
    
    # Draw layers
    boxes = []
    for x, y, w, h, label, color, details in layers:
        # Create rounded rectangle
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.5,
                            zorder=2)
        ax.add_patch(box)
        boxes.append(box)
        
        # Add text
        ax.text(x, y, label, ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=3)
    
    # Draw arrows showing data flow
    arrows = [
        # Horizontal flow
        (1.1, 5, 2.2-0.6, 0, '28×28'),
        (3.4, 5, 4.0-0.6, 0, '32×14×14'),
        (5.2, 5, 5.8-0.6, 0, '64×7×7'),
        (6.4, 5, 7.6-0.6, 0, '128×3×3'),
        (8.2, 5, 9.2-0.6, -1.5, '256'),
        # Vertical flow within blocks
        (2.2, 4.25, 0, -0.45, ''),
        (2.2, 3.4, 0, -0.6, ''),
        (4.0, 4.25, 0, -0.45, ''),
        (4.0, 3.4, 0, -0.6, ''),
        (5.8, 4.25, 0, -0.45, ''),
        (5.8, 3.4, 0, -0.6, ''),
        (7.6, 4.25, 0, -0.7, ''),
        (7.6, 2.25, 0, -0.7, ''),
        (7.6, 1.4, 0, -0.5, ''),
    ]
    
    for x, y, dx, dy, label in arrows:
        arrow = FancyArrowPatch((x, y), (x+dx, y+dy),
                               arrowstyle='->', lw=2,
                               color='#333333', zorder=1)
        ax.add_patch(arrow)
        if label:
            ax.text(x+dx/2, y+dy/2+0.2, label, ha='center',
                   fontsize=7, style='italic', color='#555555')
    
    # Add title
    ax.text(5, 9.5, 'SimpleConvNet Architecture', ha='center',
           fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input Layer'),
        mpatches.Patch(facecolor=colors['conv'], edgecolor='black', label='Convolutional Layer'),
        mpatches.Patch(facecolor=colors['pool'], edgecolor='black', label='Pooling Layer'),
        mpatches.Patch(facecolor=colors['fc'], edgecolor='black', label='Fully Connected Layer'),
        mpatches.Patch(facecolor=colors['activation'], edgecolor='black', label='Activation'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output Layer'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Add dimension annotations
    dim_annotations = [
        (1.1, 6.2, 'Input: 1×28×28'),
        (3.4, 6.2, 'After Conv1+Pool:\n32×14×14'),
        (5.2, 6.2, 'After Conv2+Pool:\n64×7×7'),
        (6.4, 6.2, 'After Conv3+Pool:\n128×3×3'),
        (7.6, 6.2, 'Flattened:\n1152'),
        (9.2, 4.5, 'Output:\n10 classes'),
    ]
    
    for x, y, text in dim_annotations:
        ax.text(x, y, text, ha='center', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('simple_conv_net_diagram.png', dpi=300, bbox_inches='tight')
    print("Diagram saved as 'simple_conv_net_diagram.png'")
    plt.show()

if __name__ == "__main__":
    visualize_simple_conv_net()

