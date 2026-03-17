from tensor import Tensor
import numpy as np

INIT_SCALE_FACTOR = 1.0  # LeCun-style initialization: sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU

# Constants for dropout
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)


# initialization of weight and bias
class Layer:
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(x, *args, **kwargs)

    def parameters(self, lr: float) -> list[Tensor]:
        return []


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, init: bool = True):
        if init:
            scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        else:
            scale = 1.0
        self.in_features = in_features
        self.out_features = out_features
        # storage should be in the shape of (out_features, in_features) for more efficient computation
        self.weight = Tensor(np.random.randn(out_features, in_features) * scale)
        self.bias = Tensor(np.random.randn(out_features))

    # forward will change the shape of x, so we need to return the new shape
    # shape of x is (batch_size, in_features)
    # shape of weight is (in_features, out_features)
    # shape of bias is (out_features,)
    # shape of output is (batch_size, out_features)
    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight.transpose()) + self.bias

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self, lr: float) -> list[Tensor]:
        return [self.weight, self.bias]


def test_unit_linear_layer():
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(10, 10))
    print(layer(x))


if __name__ == "__main__":
    test_unit_linear_layer()


class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
