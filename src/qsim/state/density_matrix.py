from __future__ import annotations

from numbers import Number, Real
from token import OP
from typing import Any
from webbrowser import Opera

import numpy as np

from qsim.lin_alg import Operator
from qsim.lin_alg.operator import OperatorLike

from .base import QuantumState, R, StateVisitor


class DensityMatrix(QuantumState):

    def __init__(self, state: np.ndarray | Operator) -> None:
        if isinstance(state, np.ndarray):
            state = Operator(state)
        if isinstance(state, Operator):
            self.state = state
        else:
            raise TypeError(
                f"Object of type {type(state)} can not be assigned to DensityMatrix.state"
            )

    def __repr__(self) -> str:
        dim = self.dim
        return f"DensityMatrix(dim={dim})"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, DensityMatrix) or isinstance(value, Operator):
            return self._operator == value._operator
        return NotImplemented

    def __call__(self, t: Real) -> DensityMatrix:
        return DensityMatrix(self._operator(t))

    def __add__(self, val: Any) -> DensityMatrix:
        if isinstance(val, DensityMatrix):
            val = val._operator
        return DensityMatrix(self._operator + val)

    def __neg__(self) -> DensityMatrix:
        return DensityMatrix(-self._operator)

    def __mul__(self, val: Number) -> DensityMatrix:
        return DensityMatrix(self._operator * val)

    def __rmul__(self, val: Number) -> DensityMatrix:
        return DensityMatrix(self._operator * val)

    def __sub__(self, val: Any) -> DensityMatrix:
        if isinstance(val, DensityMatrix):
            val = val._operator
        return DensityMatrix(self._operator - val)

    def __matmul__(self, val: Any) -> DensityMatrix:
        if isinstance(val, DensityMatrix):
            val = val._operator
        if isinstance(val, Operator):
            return DensityMatrix(self._operator @ val)
        return NotImplemented

    def __rmatmul__(self, val: Any) -> DensityMatrix:
        if isinstance(val, DensityMatrix):
            val = val._operator
        if isinstance(val, Operator):
            return DensityMatrix(val @ self._operator)
        return NotImplemented

    def __truediv__(self, val: Number) -> DensityMatrix:
        return DensityMatrix(self._operator / val)

    def __xor__(self, state: DensityMatrix) -> DensityMatrix:
        return self.tensor(state)

    def hConj(self) -> OperatorLike:
        return DensityMatrix(self._operator.hConj())

    def tensor(self, matrix: OperatorLike) -> DensityMatrix:
        if isinstance(matrix, DensityMatrix):
            matrix = matrix._operator
        return DensityMatrix(self._operator.tensor(matrix))

    def commutator(self, matrix: OperatorLike) -> OperatorLike:
        if isinstance(matrix, DensityMatrix):
            matrix = matrix._operator
        return DensityMatrix(self._operator.commutator(matrix))

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None,
    ) -> OperatorLike:
        return DensityMatrix(
            self._operator.changeHilbertSpace(new_dims, send_to_sites, base_dims)
        )

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> DensityMatrix:
        return DensityMatrix(self._operator.partialTrace(dims, reduce_to_sites))

    @property
    def dim(self) -> int:
        return self._operator.dim

    @property
    def state(self) -> np.ndarray:
        return self._operator.matrix

    @state.setter
    def state(self, state: np.ndarray | Operator) -> None:
        if isinstance(state, np.ndarray):
            state = Operator(state)
        self._operator: Operator = state

    @property
    def matrix(self) -> np.ndarray:
        return self.state

    @matrix.setter
    def matrix(self, state: np.ndarray | Operator) -> None:
        self.state = state

    def accept(self, visitor: StateVisitor[R], **kwargs) -> R:
        return visitor.visitDensityMatrix(self, **kwargs)

    def purity(self) -> float:
        return np.trace(self.state @ self.state)

    def normalise(self) -> DensityMatrix:
        return DensityMatrix(self.matrix / np.trace(self.matrix))

    def isLegitimate(self) -> bool:
        return (
            self.isSelfAdjoint()
            and self.isSemiPositive()
            and np.isclose(self.trace(), 1)
        )

    def isSelfAdjoint(self) -> bool:
        return self._operator.isSelfAdjoint()

    def isSemiPositive(self) -> bool:
        return self._operator.isSemiPositive()

    def trace(self) -> Real:
        return np.trace(self.matrix)

    def changeBasis(self, basis: np.ndarray) -> DensityMatrix:
        return DensityMatrix(self._operator.changeBasis(basis))
