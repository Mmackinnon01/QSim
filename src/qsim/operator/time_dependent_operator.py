from __future__ import annotations

from functools import reduce
from numbers import Number, Real
from tkinter import TOP
from token import OP
from turtle import TPen
from typing import Any, Callable

import numpy as np

from .base import Operator, OperatorLike


class TOperator(OperatorLike):

    def __init__(
        self, terms: list[tuple[Callable[[Real], Number], OperatorLike]]
    ) -> None:
        if len(terms) == 0:
            raise ValueError("TOperator can't be instantiated with an empty terms list")
        self._terms = terms

    def __call__(self, t: Real) -> Operator:
        if isinstance(t, Real):
            return reduce(lambda x, y: x + y, [f(t) * op(t) for (f, op) in self._terms])
        raise TypeError(f"OperatorLike protocol requires real input, not {type(t)}")

    @property
    def dim(self) -> int:
        return self._terms[0][1].matrix.shape[0]

    @classmethod
    def from_static(cls, op: Operator) -> TOperator:
        return cls([(lambda t: 1, op)])

    def __mul__(self, val: Any) -> TOperator:
        if isinstance(val, Number):
            return TOperator([(f, val * op) for f, op in self._terms])
        elif callable(val):
            if doesCallableReturnNumber(val):
                return TOperator(
                    [(lambda t: val(t) * f(t), op) for f, op in self._terms]
                )
        return NotImplemented

    def __truediv__(self, val: Number) -> OperatorLike:
        if isinstance(val, Number):
            return TOperator([(f, op / val) for f, op in self._terms])
        return NotImplemented

    def __rmul__(self, val: Any) -> TOperator:
        return self.__mul__(val)

    def __add__(self, val: Any) -> TOperator:
        if isinstance(val, Number):
            return TOperator(
                self._terms + [(lambda t: val, Operator(np.eye(self.dim)))]
            )
        elif isinstance(val, Operator):
            return self + TOperator.from_static(val)
        elif isinstance(val, TOperator):
            return TOperator(self._terms + val._terms)
        elif callable(val):
            if doesCallableReturnNumber(val):
                return TOperator(
                    self._terms + [(lambda t: val(t), Operator(np.eye(self.dim)))]
                )
        return NotImplemented

    def __radd__(self, val: Any) -> TOperator:
        return self.__add__(val)

    def __sub__(self, val: Any) -> TOperator:
        if isinstance(val, Number):
            return TOperator(
                self._terms + [(lambda t: -val, Operator(np.eye(self.dim)))]
            )
        elif isinstance(val, Operator):
            return self - TOperator.from_static(val)
        elif isinstance(val, TOperator):
            return TOperator(self._terms + [(f, -op) for f, op in val._terms])
        elif callable(val):
            if doesCallableReturnNumber(val):
                return TOperator(
                    self._terms + [(lambda t: -val(t), Operator(np.eye(self.dim)))]
                )
        return NotImplemented

    def __rsub__(self, val: Any) -> TOperator:
        return -self.__sub__(val)

    def __neg__(self) -> TOperator:
        return TOperator([(f, -op) for f, op in self._terms])

    def __matmul__(self, val: Any) -> TOperator:
        if isinstance(val, Operator):
            return TOperator([(f, op @ val) for f, op in self._terms])
        elif isinstance(val, TOperator):
            terms = []
            for f, op_left in self._terms:
                for g, op_right in val._terms:
                    terms.append((lambda t, f=f, g=g: f(t) * g(t), op_left @ op_right))
            return TOperator(terms)
        return NotImplemented

    def __rmatmul__(self, val: Any) -> TOperator:
        if isinstance(val, Operator):
            return TOperator([(f, val @ op) for f, op in self._terms])
        elif isinstance(val, TOperator):
            terms = []
            for f, op_left in val._terms:
                for g, op_right in self._terms:
                    terms.append((lambda t, f=f, g=g: f(t) * g(t), op_left @ op_right))
            return TOperator(terms)
        return NotImplemented

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> TOperator:
        embeded_terms = []

        for f, op in self._terms:
            embeded_op = op.changeHilbertSpace(new_dims, send_to_sites, base_dims)
            embeded_terms.append((f, embeded_op))

        return TOperator(embeded_terms)

    def hConj(self) -> TOperator:
        return TOperator(
            [(lambda t, f=f: f(t).conjugate(), op.hConj()) for f, op in self._terms]
        )

    def tensor(self, matrix: OperatorLike) -> OperatorLike:
        if isinstance(matrix, Operator):
            return TOperator([(f, op.tensor(matrix)) for f, op in self._terms])
        elif isinstance(matrix, TOperator):
            terms = []
            for f, op_left in self._terms:
                for g, op_right in matrix._terms:
                    if isinstance(op_left, Operator):
                        op_left = TOperator.from_static(op_left)
                    terms.append(
                        (lambda t, f=f, g=g: f(t) * g(t), op_left.tensor(op_right))
                    )
            return TOperator(terms)

    def commutator(self, matrix: OperatorLike) -> OperatorLike:
        return self @ matrix - matrix @ self


def doesCallableReturnNumber(c: Callable):
    try:
        output = c(0)
        if isinstance(output, Number):
            return True
    except Exception as e:
        pass
    return False
