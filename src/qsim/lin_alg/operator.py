from __future__ import annotations

from math import prod
from numbers import Number, Real
from token import OP
from typing import Any, Protocol, Self

import numpy as np


class OperatorLike(Protocol):
    def __call__(self, t: Real) -> OperatorLike: ...
    def __matmul__(self, val: Any) -> OperatorLike: ...
    def __rmatmul__(self, val: Any) -> OperatorLike: ...
    def __add__(self, val: Any) -> OperatorLike: ...
    def __truediv__(self, val: Number) -> OperatorLike: ...
    def __mul__(self, val: Number) -> OperatorLike: ...
    def __rmul__(self, val: Number) -> OperatorLike: ...
    def __sub__(self, val: Operator) -> OperatorLike: ...
    def __neg__(self) -> OperatorLike: ...

    @property
    def dim(self) -> int: ...
    def hConj(self) -> OperatorLike: ...
    def tensor(self, matrix: OperatorLike) -> OperatorLike: ...
    def commutator(self, matrix: OperatorLike) -> OperatorLike: ...
    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None,
    ) -> OperatorLike: ...
    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> OperatorLike: ...


class Operator(OperatorLike):

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self._eigvals = None
        self._eigvecs = None

    def __call__(self, t: Real) -> Operator:
        if isinstance(t, Real):
            return self
        raise TypeError(f"OperatorLike protocol requires real input, not {type(t)}")

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, type(self)):
            return False
        return np.allclose(self.matrix, value.matrix)

    def __repr__(self) -> str:
        dim = self.dim
        return f"Operator(dim={dim})"

    def __truediv__(self, val: Number) -> Self:
        if isinstance(val, Number):
            return type(self)(self.matrix / val)
        return NotImplemented

    def __mul__(self, val: Number) -> Self:
        if isinstance(val, Number):
            return type(self)(self.matrix * val)
        return NotImplemented

    def __rmul__(self, val: Number) -> Self:
        if isinstance(val, Number):
            return type(self)(self.matrix * val)
        return NotImplemented

    def __add__(self, val: Any) -> Self:
        if isinstance(val, Operator):
            return type(self)(self.matrix + val.matrix)
        elif isinstance(val, Number):
            return type(self)(self.matrix + val)
        return NotImplemented

    def __radd__(self, val: Any) -> Self:
        if isinstance(val, Operator):
            return type(self)(self.matrix + val.matrix)
        elif isinstance(val, Number):
            return type(self)(self.matrix + val)
        return NotImplemented

    def __sub__(self, val: Operator) -> Self:
        if isinstance(val, Operator):
            return type(self)(self.matrix - val.matrix)
        elif isinstance(val, Number):
            return type(self)(self.matrix - val)
        return NotImplemented

    def __neg__(self) -> Operator:
        return type(self)(-self.matrix)

    def __matmul__(self, matrix: Operator) -> Operator:
        if isinstance(matrix, Operator):
            return Operator(self.matrix @ matrix.matrix)
        return NotImplemented

    def __rmatmul__(self, matrix: Operator) -> Operator:
        if isinstance(matrix, Operator):
            return Operator(matrix.matrix @ self.matrix)
        return NotImplemented

    def __xor__(self, matrix: Operator) -> Operator:
        return self.tensor(matrix)

    def __pow__(self, power: int):
        if isinstance(power, int):
            return Operator(np.linalg.matrix_power(self.matrix, power))
        return NotImplemented

    @property
    def dim(self) -> int:
        return self.matrix.shape[0]

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        if not isinstance(matrix, np.ndarray):
            raise ValueError()
        self._matrix = matrix

    def conj(self) -> Self:
        return type(self)(self.matrix.conj())

    @property
    def T(self) -> Self:
        return type(self)(self.matrix.T)

    def hConj(self) -> Self:
        return type(self)(self.matrix.T.conj())

    def trace(self) -> float:
        return np.trace(self.matrix)

    def isSelfAdjoint(self) -> bool:
        return np.allclose(self.matrix.conj().T, self.matrix)

    def isSemiPositive(self) -> bool:
        return bool(np.min(np.linalg.eigvals(self.matrix)) >= -0.0001)

    def tensor(self, matrix: Self) -> Self:
        if isinstance(matrix, type(self)):
            return type(self)(np.kron(self.matrix, matrix.matrix))
        return NotImplemented

    def commutator(self, matrix: M) -> M:
        return self @ matrix - matrix @ self

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> Operator:
        """
        Embed an operator defined on a tensor-product Hilbert space
        into a larger Hilbert space.

        Parameters
        ----------
        new_dims : tuple[int]
            Dimensions of full Hilbert space.
        send_to_sites : tuple[int]
            Indices in new_dims where old subsystems are mapped.
        old_dims : tuple[int] | None
            Dimensions of subsystems the operator acts on. Defaults to (self.dim,) if None
        """

        A = self.matrix
        if base_dims is None:
            base_dims = (A.shape[0],)

        targets = tuple(send_to_sites)
        N = len(new_dims)
        k = len(targets)

        if len(base_dims) != k:
            raise ValueError("base_dims and send_to_sites must have same length")
        if len(set(targets)) != k or any(i < 0 or i >= N for i in targets):
            raise ValueError("invalid send_to_sites")
        if any(new_dims[s] != d for s, d in zip(targets, base_dims)):
            raise ValueError("dimension mismatch")
        if A.shape != (int(np.prod(base_dims)), int(np.prod(base_dims))):
            raise ValueError("operator shape incompatible with base_dims")

        rest = tuple(i for i in range(N) if i not in targets)
        order = targets + rest  # reordered subsystem order
        dims_ordered = tuple(new_dims[i] for i in order)

        d_rest = int(np.prod([new_dims[i] for i in rest])) if rest else 1
        A_ext = np.kron(A, np.eye(d_rest, dtype=A.dtype))  # acts on [targets, rest]

        # Convert from [targets, rest] basis ordering back to original site ordering.
        inv = np.argsort(order)  # original index -> position in `order`
        T = A_ext.reshape(*dims_ordered, *dims_ordered)
        axes = tuple(inv) + tuple(i + N for i in inv)
        T = T.transpose(axes)

        D = int(np.prod(new_dims))
        return Operator(T.reshape(D, D))

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> Operator:
        """
        Compute partial trace of an operator over all subsystems
        except those listed in `reduce_to_sites`, including reordering of the subsystems to match the ordering of reduce_to_sites.

        Parameters
        ----------
        op : np.ndarray
            Square matrix of shape (D, D).
        dims : sequence of int
            Hilbert space dimensions of each subsystem.
        reduce_to_sites : iterable of int
            Indices of subsystems to retain.

        Returns
        -------
        np.ndarray
            Reduced density matrix / operator.
        """

        dims = tuple(dims)
        keep = reduce_to_sites
        op = self.matrix
        n = len(dims)

        D = np.prod(dims)
        if op.shape != (D, D):
            raise ValueError("Operator shape incompatible with dims")

        # Sites to trace out
        trace = tuple(i for i in range(n) if i not in keep)

        # Reshape into tensor with 2n indices
        reshaped = op.reshape(*dims, *dims)

        # Permutation: keep (row), trace (row), keep (col), trace (col)
        perm = keep + trace + tuple(i + n for i in keep) + tuple(i + n for i in trace)

        reshaped = reshaped.transpose(perm)

        # Compute dimensions
        dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
        dim_trace = int(np.prod([dims[i] for i in trace])) if trace else 1

        reshaped = reshaped.reshape(dim_keep, dim_trace, dim_keep, dim_trace)

        # Trace over traced subsystems
        reduced = np.trace(reshaped, axis1=1, axis2=3)

        return Operator(reduced)

    @property
    def eigenvectors(self):
        if self._eigvecs is None:
            self._eigvals, self._eigvecs = np.linalg.eigh(self.matrix)
        return self._eigvecs

    @property
    def eigenvalues(self):
        if self._eigvals is None:
            self._eigvals, self._eigvecs = np.linalg.eigh(self.matrix)
        return self._eigvals

    def changeBasis(self, basis: np.ndarray) -> Operator:
        return Operator(basis).hConj() @ self @ Operator(basis)


class TestOperator(Operator):

    def __matmul__(self, matrix):
        if isinstance(matrix, TestOperator):
            return TestOperator(self.matrix @ matrix.matrix)
        return NotImplemented

    def __rmatmul__(self, matrix):
        if isinstance(matrix, TestOperator):
            return TestOperator(matrix.matrix @ self.matrix)
        return NotImplemented

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        self._matrix = matrix


sigmaX = Operator(np.array([[0, 1], [1, 0]]))
sigmaY = Operator(np.array([[0, -1j], [1j, 0]]))
sigmaZ = Operator(np.array([[1, 0], [0, -1]]))
sigmaMinus = Operator(np.array([[0, 1], [0, 0]]))
sigmaPlus = Operator(np.array([[0, 0], [1, 0]]))


def I(d: int) -> Operator:
    return Operator(np.eye(d))
