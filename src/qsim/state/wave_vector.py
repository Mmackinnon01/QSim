from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Self

import numpy as np

from qsim.operator import Operator

from .base import R, StateVisitor
from .density_matrix import DensityMatrix


class WaveVector(ABC):

    def __init__(self, state: np.ndarray) -> None:
        self.state = state

    @abstractmethod
    def __matmul__(self, other: Any) -> Self | float | DensityMatrix: ...

    @abstractmethod
    def __rmatmul__(self, other: Any) -> Self | float | DensityMatrix: ...

    def __mul__(self, value: Number) -> Self:
        if isinstance(value, Number):
            return type(self)(self.state * value)
        return NotImplemented

    def __rmul__(self, value: Number) -> Self:
        if isinstance(value, Number):
            return type(self)(self.state * value)
        return NotImplemented

    def __add__(self, value: Any) -> Self:
        if isinstance(value, Number) or isinstance(value, WaveVector):
            return type(self)(self.state + value)
        return NotImplemented

    def __radd__(self, value: Any) -> Self:
        return self.__add__(value)

    def __truediv__(self, value: Number) -> Self:
        if isinstance(value, Number):
            return type(self)(self.state / value)
        return NotImplemented

    def __xor__(self, state: Self) -> Self:
        return self.tensor(state)

    def __neg__(self) -> Self:
        return type(self)(-self.state)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @state.setter
    def state(self, state: np.ndarray) -> None:
        self._state = state

    @abstractmethod
    def hConj(self) -> Self: ...

    def norm(self) -> float:
        return np.vdot(self.state, self.state)

    def normalise(self) -> Self:
        return self / self.norm() ** 0.5

    def isNormalised(self) -> bool:
        return bool(np.isclose(self.norm(), 1))

    @property
    def dim(self) -> int:
        return self.state.shape[0]

    def tensor(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Tensor product of type {type(self)} with type {type(other)} is not possible"
            )
        return type(self)(np.kron(self.state, other.state))

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> DensityMatrix:
        """
        Compute reduced density matrix from a pure state vector
        without constructing the full density matrix. Resulting density matrix will match the ordering or reduce_to_sites

        Parameters
        ----------
        psi : np.ndarray
            State vector of shape (D,)
        dims : sequence[int]
            Hilbert space dimensions
        reduce_to_sites : iterable[int]
            Subsystems to retain

        Returns
        -------
        np.ndarray
            Reduced density matrix
        """

        dims = tuple(dims)
        keep = tuple(reduce_to_sites)
        psi = self.state
        n = len(dims)

        D = np.prod(dims)
        if psi.shape != (D,):
            raise ValueError("State vector incompatible with dims")

        trace = tuple(i for i in range(n) if i not in keep)

        # Reshape to tensor form
        psi_tensor = psi.reshape(*dims)

        # Permute so keep subsystems come first
        perm = keep + trace
        psi_tensor = psi_tensor.transpose(perm)

        dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
        dim_trace = int(np.prod([dims[i] for i in trace])) if trace else 1

        # Reshape to matrix form
        psi_matrix = psi_tensor.reshape(dim_keep, dim_trace)

        # Compute reduced density matrix
        rho_reduced = psi_matrix @ psi_matrix.conj().T

        return DensityMatrix(rho_reduced)


class Bra(WaveVector):

    def hConj(self) -> Ket:
        return Ket(self.state.conj())

    def __matmul__(self, other: Any) -> Bra | float:
        if isinstance(other, Ket):
            return np.dot(self.state, other.state)
        elif isinstance(other, DensityMatrix):
            return Bra(self.state @ other.state)
        elif isinstance(other, Operator):
            return Bra(self.state @ other.matrix)
        return NotImplemented

    def __rmatmul__(self, other: Any) -> DensityMatrix:
        if isinstance(other, Ket):
            return DensityMatrix(np.outer(other.state, self.state))
        return NotImplemented

    def accept(self, visitor: StateVisitor[R], **kwargs) -> R:
        return visitor.visitBra(self, **kwargs)


class Ket(WaveVector):

    def hConj(self) -> Bra:
        return Bra(self.state.conj())

    def __matmul__(self, other: Any) -> DensityMatrix:
        if isinstance(other, Ket):
            return DensityMatrix(np.outer(self.state, other.state))
        return NotImplemented

    def __rmatmul__(self, other: Any) -> Ket | float:
        if isinstance(other, Ket):
            return np.inner(other.state, self.state)
        elif isinstance(other, DensityMatrix):
            return Ket(other.state @ self.state)
        elif isinstance(other, Operator):
            return Ket(other.matrix @ self.state)
        return NotImplemented

    def accept(self, visitor: StateVisitor[R], **kwargs) -> R:
        return visitor.visitKet(self, **kwargs)
