from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np

from .operator import Operator


class Vector:
    def __init__(self, matrix: np.ndarray) -> None:
        if isinstance(matrix, np.ndarray):
            if len(matrix.shape) == 2 and (
                matrix.shape[0] == 1 or matrix.shape[1] == 1
            ):
                self.matrix = matrix.astype(np.complex128)
            else:
                self.matrix = matrix.reshape(-1, 1).astype(np.complex128)
        else:
            raise TypeError(f"Cannot assign object of type {type(matrix)} to Vector")

    def __matmul__(self, other: Any) -> Operator | Vector | float:
        if isinstance(other, Operator):
            return Vector(self.matrix @ other.matrix)
        elif isinstance(other, Vector):
            val = self.matrix @ other.matrix
            if val.shape == (1, 1):
                return val[0][0]
            else:
                return Operator(val)
        return NotImplemented

    def __rmatmul__(self, other: Any) -> Operator | Vector | floatt:
        if isinstance(other, Operator):
            return Vector(other.matrix @ self.matrix)
        elif isinstance(other, Vector):
            val = other.matrix @ self.matrix
            if val.shape == (1, 1):
                return val[0][0]
            else:
                return Operator(val)

    def __repr__(self) -> str:
        dim = self.dim
        return f"{self.__class__.__name__}(dim={dim})"

    def __eq__(self, other: Vector) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return np.allclose(self.matrix, other.matrix)

    def __mul__(self, value: Number) -> Vector:
        if isinstance(value, Number):
            return Vector(self.matrix * value)
        return NotImplemented

    def __rmul__(self, value: Number) -> Vector:
        if isinstance(value, Number):
            return Vector(self.matrix * value)
        return NotImplemented

    def __add__(self, value: Any) -> Vector:
        if isinstance(value, Number):
            return Vector(self.matrix + value)
        if isinstance(value, Vector):
            return Vector(self.matrix + value.matrix)
        return NotImplemented

    def __radd__(self, value: Any) -> Vector:
        return self.__add__(value)

    def __sub__(self, value: Any) -> Vector:
        if isinstance(value, Number):
            return Vector(self.matrix - value)
        if isinstance(value, Vector):
            return Vector(self.matrix - value.matrix)
        return NotImplemented

    def __rsub___(self, value: Any) -> Vector:
        if isinstance(value, Number):
            return type(self)(value - self.matrix)
        if isinstance(value, Vector):
            return type(self)(value.matrix - self.matrix)
        return NotImplemented

    def __truediv__(self, value: Number) -> Vector:
        if isinstance(value, Number):
            return type(self)(self.matrix / value)
        return NotImplemented

    def __xor__(self, state: Vector) -> Vector:
        return self.tensor(state)

    def __neg__(self) -> Vector:
        return Vector(-self.matrix)

    def conj(self) -> Vector:
        return Vector(self.matrix.conj())

    @property
    def T(self) -> Vector:
        return Vector(self.matrix.T)

    def hConj(self) -> Vector:
        return self.T.conj()

    def norm(self) -> float:
        return np.vdot(self.matrix, self.matrix)

    def normalise(self) -> Self:
        return self / self.norm() ** 0.5

    def isNormalised(self) -> bool:
        return bool(np.isclose(self.norm(), 1))

    @property
    def dim(self) -> int:
        return self.matrix.shape[0]

    def tensor(self, other: Vector) -> Vector:
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(np.kron(self.matrix, other.matrix))
