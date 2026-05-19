from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import reduce
from numbers import Number, Real
from typing import Any, Callable, Iterator

import numba as nb
import numpy as np

from .operator import Operator, OperatorLike


@dataclass
class Function:
    f: Callable
    is_conjugated: bool = False
    
    def conj(self) -> 'Function':
        return Function(f=self.f, is_conjugated=not self.is_conjugated)

@dataclass
class FunctionList:
    funcs: list[Function] = field(default_factory=list)

    def conj(self) -> 'FunctionList':
        new_funcs = [func.conj() for func in self.funcs]
        return FunctionList(funcs=new_funcs)

    def __add__(self, func_list: 'FunctionList') -> 'FunctionList':
        return FunctionList(funcs = self.funcs + func_list.funcs)
    
    def __iter__(self) -> Iterator[Function]:
        """Allows: `for func in my_function_list:`"""
        return iter(self.funcs)

    # --- Optional but highly recommended for list wrappers ---

    def __len__(self) -> int:
        """Allows: `len(my_function_list)`"""
        return len(self.funcs)

    def __getitem__(self, index: int) -> Function:
        """Allows indexing: `my_function_list[0]`"""
        return self.funcs[index]


class TOperator(OperatorLike):

    def __init__(
        self, terms: list[tuple[Callable[[Real], Number] | FunctionList, OperatorLike]]
    ) -> None:
        if len(terms) == 0:
            raise ValueError("TOperator can't be instantiated with an empty terms list")
        self._terms = []
        for term in terms:
            if isinstance(term[0], FunctionList):
                self._terms.append(term)
            else:
                self._terms.append((FunctionList([Function(term[0])]), term[1]))

    def __call__(self, t: Real) -> Operator:
        if isinstance(t, Real):
            # 1. Evaluate terms explicitly using a standard loop
            evaluated_terms = []
            for flist, op in self._terms:
                # Python can comfortably call the jitted function f(t) here
                term = 1
                for f in flist:
                    if f.is_conjugated:
                        term *= f.f(t).conjugate()
                    else:
                        term *= f.f(t)
                evaluated_terms.append(term * op(t))
                
            # 2. Reduce the list safely
            return reduce(lambda x, y: x + y, evaluated_terms)
            
        raise TypeError(f"OperatorLike protocol requires real input, not {type(t)}")

    def __repr__(self):
        return f"TOperator(dim = {self.dim}, {len(self._terms)} terms)"

    @property
    def dim(self) -> int:
        return self._terms[0][1].matrix.shape[0]

    @classmethod
    def from_static(cls, op: Operator) -> TOperator:
        return cls([(FunctionList([Function(lambda t: 1)]), op)])

    def __mul__(self, val: Any) -> TOperator:
        if isinstance(val, Number):
            return TOperator([(fs, val * op) for fs, op in self._terms])
        elif callable(val):
            if doesCallableReturnNumber(val):
                return TOperator(
                    [(fs + FunctionList([Function(val)]), op) for fs, op in self._terms]
                )
        return NotImplemented

    def __truediv__(self, val: Number) -> TOperator:
        if isinstance(val, Number):
            return TOperator([(fs, op / val) for fs, op in self._terms])
        return NotImplemented

    def __rmul__(self, val: Any) -> TOperator:
        return self.__mul__(val)

    def __add__(self, val: Any) -> TOperator:
        if isinstance(val, Number):
            return TOperator(
                self._terms + [(FunctionList([Function(lambda t: val)]), Operator(np.eye(self.dim)))]
            )
        elif isinstance(val, Operator):
            return self + TOperator.from_static(val)
        elif isinstance(val, TOperator):
            return TOperator(self._terms + val._terms)
        elif callable(val):
            if doesCallableReturnNumber(val):
                return TOperator(
                    self._terms + [(FunctionList([Function(val)]), Operator(np.eye(self.dim)))]
                )
        return NotImplemented

    def __radd__(self, val: Any) -> TOperator:
        return self.__add__(val)

    def __sub__(self, val: Any) -> TOperator:
        if isinstance(val, Number):
            return TOperator(
                self._terms + [(FunctionList([Function(lambda t: -val)]), Operator(np.eye(self.dim)))]
            )
        elif isinstance(val, Operator):
            return self - TOperator.from_static(val)
        elif isinstance(val, TOperator):
            return TOperator(self._terms + [(fs, -op) for fs, op in val._terms])
        elif callable(val):
            if doesCallableReturnNumber(val):
                return TOperator(
                    self._terms + [(FunctionList([Function(val),Function(lambda t: -1)]), Operator(np.eye(self.dim)))]
                )
        return NotImplemented

    def __rsub__(self, val: Any) -> TOperator:
        return -self.__sub__(val)

    def __neg__(self) -> TOperator:
        return TOperator([(fs, -op) for fs, op in self._terms])

    def __matmul__(self, val: Any) -> TOperator:
        if isinstance(val, Operator):
            return TOperator([(fs, op @ val) for fs, op in self._terms])
        elif isinstance(val, TOperator):
            terms = []
            for fs, op_left in self._terms:
                for gs, op_right in val._terms:
                    terms.append((fs + gs, op_left @ op_right))
            return TOperator(terms)
        return NotImplemented

    def __rmatmul__(self, val: Any) -> TOperator:
        if isinstance(val, Operator):
            return TOperator([(fs, val @ op) for fs, op in self._terms])
        elif isinstance(val, TOperator):
            terms = []
            for fs, op_left in val._terms:
                for gs, op_right in self._terms:
                    terms.append((fs + gs, op_left @ op_right))
            return TOperator(terms)
        return NotImplemented

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> TOperator:
        embeded_terms = []

        for fs, op in self._terms:
            embeded_op = op.changeHilbertSpace(new_dims, send_to_sites, base_dims)
            embeded_terms.append((fs, embeded_op))

        return TOperator(embeded_terms)

    def hConj(self) -> TOperator:
        return TOperator(
            [(fs.conj(), op.hConj()) for fs, op in self._terms]
        )

    def conj(self) -> TOperator:
        return TOperator(
            [(fs.conj(), op.conj()) for fs, op in self._terms]
        )

    @property
    def T(self) -> TOperator:
        return TOperator([(fs, op.T) for fs, op in self._terms])

    def __xor__(self, matrix: OperatorLike) -> TOperator:
        return self.tensor(matrix)

    def __rxor__(self, matrix: OperatorLike) -> TOperator:
        if isinstance(matrix, Operator):
            operator = TOperator.from_static(matrix)
        return operator.tensor(self)

    def tensor(self, matrix: OperatorLike) -> TOperator:
        if isinstance(matrix, Operator):
            return TOperator([(fs, op.tensor(matrix)) for fs, op in self._terms])
        elif isinstance(matrix, TOperator):
            terms = []
            for fs, op_left in self._terms:
                for gs, op_right in matrix._terms:
                    terms.append(
                        (fs + gs, op_left.tensor(op_right))
                    )
            return TOperator(terms)
        return NotImplemented

    def commutator(self, matrix: OperatorLike) -> TOperator:
        return self @ matrix - matrix @ self

    def changeBasis(self, basis: np.ndarray) -> TOperator:
        return TOperator([(fs, op.changeBasis(basis)) for fs, op in self._terms])

    def partialTrace(
        self, dims: tuple[int, ...], reduce_to_sites: tuple[int, ...]
    ) -> OperatorLike:
        return TOperator(
            [(fs, op.partialTrace(dims, reduce_to_sites)) for fs, op in self._terms]
        )


    def compile(self) -> Callable[[float], np.ndarray]:
        # 1. EXTRACT DATA OUTSIDE OF NUMBA
        # Pull the static data out of 'self' so Numba doesn't have to look at objects
        dim = self.dim
        # Create a 3D NumPy array of all the matrices: shape (num_terms, dim, dim)
        matrices = np.array([op.matrix for funcs, op in self._terms], dtype=np.complex128)
        
        num_terms = len(self._terms)

        # 2. THE PURE NUMBA BACKEND (Matrix Addition)
        # This only sees raw arrays and numbers. It runs at C-speed.
        @nb.njit
        def sum_matrices(coeffs, mats):
            total = np.zeros((dim, dim), dtype=mats.dtype)
            for i in range(len(coeffs)):
                total += coeffs[i] * mats[i]
            return total

        # 3. THE HYBRID WRAPPER
        # This does the nested logic in Python, then hands off the heavy lifting
        def f(t: float) -> np.ndarray:
            # Pre-allocate an array for the evaluated scalar of each term
            coeffs = np.zeros(num_terms, dtype=np.complex128)
            
            # Standard Python loop: fast enough for scalars, allows ANY function type
            for i, (funcs, op) in enumerate(self._terms):
                partial = 1.0 + 0.0j
                for func in funcs.funcs: # Assuming funcs is your FunctionList dataclass
                    # Evaluate the raw function
                    val = func.f(t)
                    # Apply conjugation if the flag is True
                    if func.is_conjugated:
                        val = np.conjugate(val)
                    partial *= val
                    
                coeffs[i] = partial
                
            # Drop down to the Numba machine-code layer to do the NxN matrix math
            return sum_matrices(coeffs, matrices)

        return f


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
    
    def compile(self) -> Callable:
        return self._op.compile()


def doesCallableReturnNumber(c: Callable):
    try:
        output = c(0)
        if isinstance(output, Number):
            return True
    except Exception:
        pass
    return False
