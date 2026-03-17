from tensor import Tensor
import numpy as np


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

        for param in self.params:
            if isinstance(param, Tensor):
                param.requires_grad = True
                param.grad = None
        self.step_count = 0

    def step(self):
        raise NotImplementedError("")

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class _ExtractGradientMixin:
    """Mixin added to Optimizer for gradient extraction."""

    def _extract_gradient(self, param: Tensor) -> np.ndarray:
        grad = param.grad
        if isinstance(grad, Tensor):
            return grad.data
        else:
            return grad
        ### END SOLUTION


# Attach _extract_gradient to Optimizer so all subclasses inherit it
Optimizer._extract_gradient = _ExtractGradientMixin._extract_gradient


class SGD(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.momentum_buffers = [None for _ in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            # Update momentum buffer
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    # Initialize momentum buffer
                    self.momentum_buffers[i] = np.zeros_like(param.data)

                # Update momentum: v = momentum * v_prev + grad
                self.momentum_buffers[i] = (
                    self.momentum * self.momentum_buffers[i] + grad_data
                )
                grad_data = self.momentum_buffers[i]

            # Update parameter: param = param - lr * grad
            param.data = param.data - self.lr * grad_data

        # Increment step counter
        self.step_count += 1

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class Adam(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data


class _AdamWUpdateMomentsMixin:
    """Mixin added to AdamW for moment updates."""

    def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(grad_data)
            self.v_buffers[i] = np.zeros_like(grad_data)

        # Update biased first moment estimate
        self.m_buffers[i] = (
            self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
        )

        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (
            grad_data**2
        )

        # Compute bias correction
        bias_correction1 = 1 - self.beta1**self.step_count
        bias_correction2 = 1 - self.beta2**self.step_count

        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2

        return m_hat, v_hat
        ### END SOLUTION


# Attach _update_moments to AdamW
Adam._update_moments = _AdamWUpdateMomentsMixin._update_moments
