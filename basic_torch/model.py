from torch import nn
import torch
from torch import Tensor
from torch.nn import Linear, Module

def main():
    data: Tensor = torch.randn(20)
    print(data)
    linear_nn: Linear = nn.Linear(20, 10)

    print(linear_nn(data))
    print(linear_nn.weight.shape)
    print(linear_nn.bias.shape)

    relu_nn: ReLU = nn.ReLU()
    print(relu_nn(data))

    loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss()
    print(loss_fn(torch.randn(10),torch.randn(10)))

if __name__ == "__main__":
    main()