import numpy as np
from tensor import Tensor
from activation import Softmax


# compute the
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log-softmax with numerical stability.

    TODO: Implement numerically stable log-softmax using the log-sum-exp trick

    APPROACH:
    1. Find maximum along dimension (for stability)
    2. Subtract max from input (prevents overflow)
    3. Compute log(sum(exp(shifted_input)))
    4. Return input - max - log_sum_exp

    EXAMPLE:
    >>> logits = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    >>> result = log_softmax(logits, dim=-1)
    >>> print(result.shape)
    (2, 3)

    HINT: Use np.max(x.data, axis=dim, keepdims=True) to preserve dimensions
    """
    ### BEGIN SOLUTION
    # Step 1: Find max along dimension for numerical stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)
    print(max_vals)

    # Step 2: Subtract max to prevent overflow
    shifted = x.data - max_vals
    print(shifted)
    print("=" * 20)

    print(np.exp(shifted))

    # Step 3: Compute log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
    print("=" * 20)
    print(log_sum_exp)
    print("=" * 20)

    # Step 4: Return log_softmax = input - max - log_sum_exp
    result = x.data - max_vals - log_sum_exp
    print(result)

    return Tensor(result)


class MSELoss:
    def __init__(self):
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return Tensor(np.mean((predictions.data - targets.data) ** 2))

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)


logits = Tensor([[2.1, 2.1, 2.1], [2.1, 2.2, 1.9]])
result = log_softmax(logits, dim=-1)
print(result)
