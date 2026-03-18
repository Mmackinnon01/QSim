from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Self

import numpy as np

from qsim.lin_alg import Operator, Vector

from .base import R, StateVisitor
from .density_matrix import DensityMatrix


class WaveVector(ABC):

    @abstractmethod
    def __init__(self, state: np.ndarray) -> None: ...

    @abstractmethod
    def __matmul__(self, other: Any) -> Self | float | DensityMatrix: ...

    @abstractmethod
    def __rmatmul__(self, other: Any) -> Self | float | DensityMatrix: ...

    @abstractmethod
    def changeBasis(self, basis: np.ndarray) -> Self: ...

    def __repr__(self) -> str:
        dim = self.dim
        return f"{self.__class__.__name__}(dim={dim})"

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._vector == other._vector

    def __mul__(self, value: Number) -> Self:
        if isinstance(value, Number):
            return type(self)(self._vector * value)
        return NotImplemented

    def __rmul__(self, value: Number) -> Self:
        if isinstance(value, Number):
            return type(self)(self._vector * value)
        return NotImplemented

    def __add__(self, value: Any) -> Self:
        if isinstance(value, Number):
            return type(self)(self._vector + value)
        if isinstance(value, WaveVector):
            return type(self)(self._vector + value._vector)
        return NotImplemented

    def __radd__(self, value: Any) -> Self:
        return self.__add__(value)

    def __sub__(self, value: Any) -> Self:
        if isinstance(value, Number):
            return type(self)(self._vector - value)
        if isinstance(value, WaveVector):
            return type(self)(self._vector - value._vector)
        return NotImplemented

    def __rsub___(self, value: Any) -> Self:
        if isinstance(value, Number):
            return type(self)(value - self._vector)
        if isinstance(value, WaveVector):
            return type(self)(value.state - self._vector)
        return NotImplemented

    def __truediv__(self, value: Number) -> Self:
        if isinstance(value, Number):
            return type(self)(self._vector / value)
        return NotImplemented

    def __xor__(self, state: Self) -> Self:
        return self.tensor(state)

    def __neg__(self) -> Self:
        return type(self)(-self._vector)

    @property
    def state(self) -> Vector:
        return self._vector.matrix

    @state.setter
    def state(self, state: Vector) -> None:
        if isinstance(state, Vector):
            self._vector = state
        else:
            raise TypeError(
                f"Cannot set state of WaveVector to an object of type {type(state)}"
            )

    @property
    def matrix(self) -> np.ndarray:
        return self.state

    @abstractmethod
    def hConj(self) -> Self: ...

    def norm(self) -> float:
        return self._vector.norm()

    def normalise(self) -> Self:
        return self / self.norm() ** 0.5

    def isNormalised(self) -> bool:
        return bool(np.isclose(self.norm(), 1))

    @property
    def dim(self) -> int:
        return self._vector.dim

    def tensor(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)(self._vector ^ other._vector)

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
        psi = self.matrix
        n = len(dims)

        D = np.prod(dims)
        if psi.shape != (D, 1):
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

    def __init__(self, state: np.ndarray) -> None:
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            if state.shape[0] == 1:
                state = Vector(state)
            else:
                raise ValueError(
                    f"Array of shape {state.shape} not valid for Bra, must be (1, -1)"
                )
        if isinstance(state, Vector):
            self.state = state
        else:
            raise TypeError(f"Cannot assign object of type {type(state)} to Bra")

    def hConj(self) -> Ket:
        return Ket(self._vector.hConj())

    def __matmul__(self, other: Any) -> Bra | Number:
        if isinstance(other, Ket):
            return self._vector @ other._vector
        elif isinstance(other, DensityMatrix):
            return Bra(self._vector @ other._operator)
        elif isinstance(other, Operator):
            return Bra(self._vector @ other)
        return NotImplemented

    def __rmatmul__(self, other: Any) -> DensityMatrix:
        if isinstance(other, Ket):
            return DensityMatrix(other._vector @ self._vector)
        return NotImplemented

    def accept(self, visitor: StateVisitor[R], **kwargs) -> R:
        return visitor.visitBra(self, **kwargs)

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> DensityMatrix:
        ket_self = self.hConj()
        return (ket_self.partialTrace(dims, reduce_to_sites)).hConj()

    def changeBasis(self, basis: np.ndarray) -> Bra:
        return self @ Operator(basis)


class Ket(WaveVector):

    def __init__(self, state: np.ndarray) -> None:
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                state = state.reshape(-1, 1)
            if state.shape[1] == 1:
                state = Vector(state)
            else:
                raise ValueError(
                    f"Array of shape {state.shape} not valid for Ket, must be (-1, 1)"
                )
        if isinstance(state, Vector):
            self.state = state
        else:
            raise TypeError(f"Cannot assign object of type {type(state)} to Ket")

    def hConj(self) -> Bra:
        return Bra(self._vector.hConj())

    def __matmul__(self, other: Any) -> DensityMatrix:
        if isinstance(other, Ket):
            return DensityMatrix(self.matrix @ other.matrix)
        return NotImplemented

    def __rmatmul__(self, other: Any) -> Ket | float:
        if isinstance(other, Ket):
            return other._vector @ self._vector
        elif isinstance(other, DensityMatrix):
            return Ket(other._operator @ self._vector)
        elif isinstance(other, Operator):
            return Ket(other @ self._vector)
        return NotImplemented

    def accept(self, visitor: StateVisitor[R], **kwargs) -> R:
        return visitor.visitKet(self, **kwargs)

    def changeBasis(self, basis: np.ndarray) -> Ket:
        return Operator(basis).hConj() @ self
