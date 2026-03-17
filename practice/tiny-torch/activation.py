import torch
from tensor import Tensor
import numpy as np


class Sigmoid:
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))

    def parameters(self):
        return []


class Relu:
    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0, x.data)
        return Tensor(result)

    def parameters(self, x):
        return []


class Tanh:
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.tanh(x.data))

    def parameters(self):
        return []


class Softmax:
    def __init__(self):
        pass

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max

        # Compute exponentials
        exp_values = np.exp(x_shifted)

        # Sum along dimension
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return Tensor(result)

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        return self.forward(x, dim)

    def parameters(self):
        return []


class GELU:
    def forward(self, x: Tensor) -> Tensor:
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        # Then multiply by x
        result = x.data * sigmoid_part
        return Tensor(result)

    def parameters(self):
        return []


def test_unit_softmax():
    softmax = Softmax()
    logits = Tensor([[2.1, 2.1, 2.1], [2.1, 2.2, 1.9]])
    result = softmax(logits, dim=-1)
    print(result)


if __name__ == "__main__":
    test_unit_softmax()
