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

    def __call__(self, t: Real) -> Operator:
        if isinstance(t, Real):
            return self
        raise TypeError(f"OperatorLike protocol requires real input, not {type(t)}")

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, type(self)):
            return False
        return np.allclose(self.matrix, value.matrix)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"{np.array2string(self.matrix, precision=3)}\n)"
        )

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
        raise TypeError(
            "Tensor product only possible with two matrices of the same type"
        )

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

        if not base_dims:
            base_dims = (self.dim,)

        if len(base_dims) != len(send_to_sites):
            raise ValueError("old_dims and send_to_sites must have same length")

        if any(site >= len(new_dims) or site < 0 for site in send_to_sites):
            raise ValueError("send_to_sites contains invalid index")

        if len(set(send_to_sites)) != len(send_to_sites):
            raise ValueError("send_to_sites contains duplicate index")

        # Check dimension compatibility
        for old_dim, site in zip(base_dims, send_to_sites):
            if new_dims[site] != old_dim:
                raise ValueError(
                    f"Dimension mismatch at site {site}: "
                    f"{new_dims[site]} != {old_dim}"
                )

        N = len(new_dims)

        # --- Step 1: reshape operator into tensor form ---
        # shape: (d0,...,dk-1, d0,...,dk-1)
        op_tensor = self.matrix.reshape(*base_dims, *base_dims)

        # --- Step 2: Build full tensor product space ---
        # First create identity on untouched subsystems
        full_dim = prod(new_dims)
        full_tensor = np.eye(full_dim, dtype=self.matrix.dtype).reshape(
            *new_dims, *new_dims
        )

        # Build index mapping
        for left_indices in np.ndindex(*new_dims):
            for right_indices in np.ndindex(*new_dims):
                if all(
                    left_indices[i] == right_indices[i]
                    for i in range(N)
                    if i not in send_to_sites
                ):
                    old_left = tuple(left_indices[i] for i in send_to_sites)
                    old_right = tuple(right_indices[i] for i in send_to_sites)
                    full_tensor[(*left_indices, *right_indices)] = op_tensor[
                        (*old_left, *old_right)
                    ]

        return Operator(full_tensor.reshape(full_dim, full_dim))

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> Operator:
        """
        Compute partial trace of an operator over all subsystems
        except those listed in `keep`.

        Parameters
        ----------
        op : np.ndarray
            Square matrix of shape (D, D).
        dims : sequence of int
            Hilbert space dimensions of each subsystem.
        keep : iterable of int
            Indices of subsystems to retain.

        Returns
        -------
        np.ndarray
            Reduced density matrix / operator.
        """

        dims = tuple(dims)
        keep = tuple(sorted(reduce_to_sites))
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
