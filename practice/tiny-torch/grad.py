from tensor import Tensor


class Function:
    def __init__(self, *tensors: Tensor):
        self.saved_tensors = tensors

    def apply(self, grad_output: Tensor):
        raise NotImplementedError("Subclasses must implement apply method")


def enable_autograd(quiet=False):
    _original_add = Tensor.__add__

    def tracked_add(self, other: Self | float | int) -> Self:
        print(f"extend_add is called: {self} + {other}")
        result = _original_add(self, other)
        result.extension = AddExtension()
        return result


class AddBackward(Function):
    def apply(self, grad_output: Tensor):
        print(f"AddExtension apply is called: {grad_output}")


a = Tensor(np.array([1, 2, 3]))
b = Tensor(np.array([4, 5, 6]))
c = a + b
print(c.extension.apply(Tensor(np.array([1, 1, 1]))))


class Add(Function):
    def apply(self, grad_output: Tensor):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output

        # Gradient for second input
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output

        return grad_a, grad_b


Tensor.__add__ = extend_add
