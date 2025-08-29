from torch import nn
import torch

def main():
    data = torch.randn(20)
    print(data)
    linear_nn = nn.Linear(20, 10)

    print(linear_nn(data))
    print(linear_nn.weight.shape)
    print(linear_nn.bias.shape)

    relu_nn = nn.ReLU()
    print(relu_nn(data))

    loss_fn = nn.CrossEntropyLoss()
    print(loss_fn(torch.randn(10),torch.randn(10)))

if __name__ == "__main__":
    main()