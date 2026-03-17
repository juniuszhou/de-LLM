# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %%
from typing import Self
import numpy as np


class Tensor:
    data: np.ndarray

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __add__(self, other: Self | float | int) -> Self:
        print(f"I am original add")
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other: Self | float | int) -> Self:
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __repr__(self):
        """String representation of tensor for debugging."""
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """Human-readable string representation."""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data

    def memory_footprint(self):
        return self.data.nbytes

    def __mul__(self, other: Self | float | int) -> Self:
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other: Self | float | int) -> Self:
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def __pow__(self, other: Self | float | int) -> Self:
        if isinstance(other, Tensor):
            return Tensor(self.data**other.data)
        else:
            return Tensor(self.data**other)

    def __mod__(self, other: Self | float | int) -> Self:
        if isinstance(other, Tensor):
            return Tensor(self.data % other.data)
        else:
            return Tensor(self.data % other)

    def matmul(self, other: Self) -> Self:
        return Tensor(np.matmul(self.data, other.data))

    def reshape(self, shape: tuple[int, ...]) -> Self:
        return Tensor(self.data.reshape(shape))

    def transpose(self, axes: tuple[int, ...] | None = None) -> Self:
        return Tensor(self.data.transpose(axes))

    def flatten(self) -> Self:
        return Tensor(self.data.flatten())

    def sum(self, axis: int | None = None) -> Self:
        return Tensor(self.data.sum(axis))

    def mean(self, axis: int | None = None) -> Self:
        return Tensor(self.data.mean(axis))


# %%
a = Tensor(np.array([1, 2, 3]))
