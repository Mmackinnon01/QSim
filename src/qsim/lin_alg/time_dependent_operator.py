from __future__ import annotations

import bisect
from collections import OrderedDict
from functools import reduce
from numbers import Number, Real
from typing import Any, Callable

import numpy as np

from .operator import Operator, OperatorLike


class TOperator(OperatorLike):

    def __init__(
        self, terms: list[tuple[Callable[[Real], Number], OperatorLike]]
    ) -> None:
        if len(terms) == 0:
            raise ValueError("TOperator can't be instantiated with an empty terms list")
        self._terms = []
        for term in terms:
            self._terms.append(term)

    def __call__(self, t: Real) -> Operator:
        if isinstance(t, Real):
            return reduce(lambda x, y: x + y, [f(t) * op(t) for (f, op) in self._terms])
        raise TypeError(f"OperatorLike protocol requires real input, not {type(t)}")

    def __repr__(self):
        term_strs = []
        for f, op in self._terms:
            fname = getattr(f, "__name__", f.__class__.__name__)
            term_strs.append(f"{fname}(t) * {repr(op)}")
        inner = ",\n  ".join(term_strs)
        return f"TOperator(\n  {inner}\n)"

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

    def __truediv__(self, val: Number) -> TOperator:
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

    def conj(self) -> TOperator:
        return TOperator(
            [(lambda t, f=f: f(t).conjugate(), op.conj()) for f, op in self._terms]
        )

    @property
    def T(self) -> TOperator:
        return TOperator([(lambda t, f=f: f(t), op.T) for f, op in self._terms])

    def __xor__(self, matrix: OperatorLike) -> TOperator:
        return self.tensor(matrix)

    def __rxor__(self, matrix: OperatorLike) -> TOperator:
        if isinstance(matrix, Operator):
            operator = TOperator.from_static(matrix)
        return operator.tensor(self)

    def tensor(self, matrix: OperatorLike) -> TOperator:
        if isinstance(matrix, Operator):
            return TOperator([(f, op.tensor(matrix)) for f, op in self._terms])
        elif isinstance(matrix, TOperator):
            terms = []
            for f, op_left in self._terms:
                for g, op_right in matrix._terms:
                    terms.append(
                        (lambda t, f=f, g=g: f(t) * g(t), op_left.tensor(op_right))
                    )
            return TOperator(terms)
        return NotImplemented

    def commutator(self, matrix: OperatorLike) -> TOperator:
        return self @ matrix - matrix @ self

    def changeBasis(self, basis: np.ndarray) -> TOperator:
        return TOperator([(f, op.changeBasis(basis)) for f, op in self._terms])

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> OperatorLike:
        return TOperator(
            [(f, op.partialTrace(dims, reduce_to_sites)) for f, op in self._terms]
        )


class DiscreteTOperator(OperatorLike):

    def __init__(self, op: TOperator, intervals: tuple, cache_length: int = 20) -> None:
        self._op: TOperator = op
        self._intervals: tuple = tuple(sorted(intervals))
        self._cache: OrderedDict = OrderedDict()
        self._cache_length: int = cache_length

    def __call__(self, t: Real) -> Operator:
        interval = self._getInterval(t)
        if interval not in self._cache:
            self._cache[interval] = self._op(t)
            if len(self._cache) > self._cache_length:
                self._cache.popitem(last=False)
        return self._cache[interval]

    def _getInterval(self, t: Real) -> tuple:
        if t < self._intervals[0]:
            return (-1, self._intervals[0])
        elif t > self._intervals[-1]:
            return (self._intervals[-1], -1)
        else:
            i = bisect.bisect_left(self._intervals, t)
            return self._intervals[i : i + 2]

    @property
    def dim(self) -> int:
        return self._op.dim

    def __repr__(self) -> str:
        return super().__repr__()

    def __matmul__(self, matrix: OperatorLike) -> OperatorLike:
        if isinstance(matrix, Operator):
            return DiscreteTOperator(self._op @ matrix, self._intervals)
        elif isinstance(matrix, TOperator):
            return self._op @ matrix
        elif isinstance(matrix, DiscreteTOperator):
            return DiscreteTOperator(
                self._op @ matrix._op,
                intervals=tuple(sorted(set(self._intervals) | set(matrix._intervals))),
            )

    def __rmatmul__(self, matrix: OperatorLike) -> OperatorLike:
        if isinstance(matrix, Operator):
            return DiscreteTOperator(matrix @ self._op, self._intervals)
        elif isinstance(matrix, TOperator):
            return matrix @ self._op
        elif isinstance(matrix, DiscreteTOperator):
            return DiscreteTOperator(
                matrix._op @ self._op,
                intervals=tuple(sorted(set(self._intervals) | set(matrix._intervals))),
            )

    def __add__(self, val: Any) -> OperatorLike:
        if isinstance(val, TOperator):
            return self._op + val
        elif isinstance(val, DiscreteTOperator):
            return DiscreteTOperator(
                val._op + self._op,
                intervals=tuple(sorted(set(self._intervals) | set(val._intervals))),
            )
        else:
            return DiscreteTOperator(self._op + val, self._intervals)

    def __radd__(self, val: Any) -> OperatorLike:
        return self.__add__(val)

    def __sub__(self, val: Any) -> OperatorLike:
        if isinstance(val, TOperator):
            return self._op - val
        elif isinstance(val, DiscreteTOperator):
            return DiscreteTOperator(
                val._op - self._op,
                intervals=tuple(sorted(set(self._intervals) | set(val._intervals))),
            )
        else:
            return DiscreteTOperator(self._op - val, self._intervals)

    def __rsub__(self, val: Any) -> OperatorLike:
        if isinstance(val, TOperator):
            return val - self._op
        elif isinstance(val, DiscreteTOperator):
            return DiscreteTOperator(
                self._op - val._op,
                tuple(sorted(set(self._intervals) | set(val._intervals))),
            )
        else:
            return DiscreteTOperator(val - self._op, self._intervals)

    def __mul__(self, val: Number) -> DiscreteTOperator:
        return DiscreteTOperator(self._op * val, self.intervals)

    def __rmul__(self, val: Number) -> DiscreteTOperator:
        return DiscreteTOperator(val * self._op, self.intervals)

    def __truediv__(self, val: Number) -> DiscreteTOperator:
        return DiscreteTOperator(self._op / val, self._intervals)

    def __xor__(self, matrix: OperatorLike) -> OperatorLike:
        if isinstance(matrix, TOperator):
            return self._op ^ matrix
        elif isinstance(matrix, DiscreteTOperator):
            return DiscreteTOperator(
                self._op ^ matrix._op,
                tuple(sorted(set(self._intervals) | set(matrix._intervals))),
            )
        else:
            return DiscreteTOperator(self._op ^ matrix, self._intervals)

    def __rxor__(self, matrix):
        if isinstance(matrix, TOperator):
            return matrix ^ self._op
        elif isinstance(matrix, DiscreteTOperator):
            return DiscreteTOperator(
                matrix._op ^ self._op,
                tuple(sorted(set(self._intervals) | set(matrix._intervals))),
            )
        else:
            return DiscreteTOperator(matrix ^ self._op, self._intervals)

    def __neg__(self) -> DiscreteTOperator:
        return DiscreteTOperator(-self._op, self._intervals)

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> DiscreteTOperator:
        return DiscreteTOperator(
            self._op.changeHilbertSpace(new_dims, send_to_sites, base_dims),
            self._intervals,
        )

    def hConj(self) -> DiscreteTOperator:
        return DiscreteTOperator(self._op.hConj(), self._intervals)

    def conj(self) -> DiscreteTOperator:
        return DiscreteTOperator(self._op.conj(), self._intervals)

    @property
    def T(self) -> DiscreteTOperator:
        return DiscreteTOperator(self._op.T, self._intervals)

    def tensor(self, matrix: OperatorLike) -> OperatorLike:
        return self ^ matrix

    def commutator(self, matrix: OperatorLike) -> OperatorLike:
        return self @ matrix - matrix @ self

    def changeBasis(self, basis: np.ndarray) -> DiscreteTOperator:
        return DiscreteTOperator(self._op.changeBasis(basis), self._intervals)

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> DiscreteTOperator:
        return DiscreteTOperator(self._op.partialTrace, self._intervals)


def doesCallableReturnNumber(c: Callable):
    try:
        output = c(0)
        if isinstance(output, Number):
            return True
    except Exception:
        pass
    return False
