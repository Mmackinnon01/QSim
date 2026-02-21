from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from numbers import Real
from typing import Self

from scipy.linalg import expm

from qsim.operator import Operator
from qsim.operator.base import OperatorLike
from qsim.state import Bra, DensityMatrix, Ket, QuantumState, StateVisitor


class Generator(ABC, StateVisitor):

    @abstractmethod
    def __add__(self, generator: Self) -> Self: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def visitKet(self, psi: Ket, t: Real) -> Ket: ...

    @abstractmethod
    def visitBra(self, psi: Bra, t: Real) -> Bra: ...

    @abstractmethod
    def visitDensityMatrix(self, rho: DensityMatrix, t: Real) -> DensityMatrix: ...

    @abstractmethod
    def onOperator(self, op: Operator, t: Real = 0) -> Operator: ...

    def onState(self, state: QuantumState, t: Real = 0) -> QuantumState:
        return state.accept(self, t=t)

    @abstractmethod
    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> Self: ...


class GKSLGenerator(Generator):
    def __init__(
        self, H: OperatorLike, jumps: list[OperatorLike] | None = None
    ) -> None:
        if jumps:
            self.jumps = jumps
        else:
            self.jumps = []

        self.H = H

    def __add__(self, dynamic: GKSLGenerator) -> GKSLGenerator:
        if isinstance(dynamic, GKSLGenerator):
            if self.dim == dynamic.dim:
                return GKSLGenerator(
                    self.H + dynamic.H, jumps=self.jumps + dynamic.jumps
                )
            else:
                raise ValueError(
                    f"Cannot add UnitaryDynamics with dims={self.dim}, {dynamic.dim}"
                )
        return NotImplemented

    @property
    def dim(self) -> int:
        return self.H.dim

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> GKSLGenerator:
        if not base_dims:
            base_dims = (self.dim,)
        embedded_H = self.H.changeHilbertSpace(new_dims, send_to_sites, base_dims)
        embedded_jumps = [
            jump.changeHilbertSpace(new_dims, send_to_sites, base_dims)
            for jump in self.jumps
        ]
        return GKSLGenerator(embedded_H, embedded_jumps)

    def onOperator(self, op: Operator, t: Real = 0) -> Operator:
        H, jumps, hJumps, ac_components = self._evaluateOperators(t)
        unitary_component = 1j * (H @ op - op @ H)
        if len(jumps) == 0:
            return unitary_component
        dissipative_component = reduce(
            lambda x, y: x + y,
            [
                L_dag @ op @ L - 0.5 * (L2 @ op + op @ L2)
                for L, L_dag, L2 in zip(jumps, hJumps, ac_components)
            ],
        )
        return unitary_component + dissipative_component

    def visitDensityMatrix(self, rho: DensityMatrix, t: Real) -> DensityMatrix:
        H, jumps, hJumps, ac_components = self._evaluateOperators(t)
        unitary_component = -1j * (H @ rho - rho @ H)
        if len(self.jumps) == 0:
            return unitary_component
        dissipative_component = reduce(
            lambda x, y: x + y,
            [
                L @ rho @ L_dag - 0.5 * (L2 @ rho + rho @ L2)
                for L, L_dag, L2 in zip(jumps, hJumps, ac_components)
            ],
        )
        return (unitary_component + dissipative_component)(t)

    def visitBra(self, psi: Bra, t: float) -> TypeError:
        raise TypeError("GKSL master equation not valid for wavevector input")

    def visitKet(self, psi: Ket, t: float) -> TypeError:
        raise TypeError("GKSL master equation not valid for wavevector input")

    def _evaluateOperators(self, t: Real):
        H = self.H(t)
        jumps = [jump(t) for jump in self.jumps]
        hconj_jumps = [jump.hConj() for jump in jumps]
        anticommutator_components = [
            jump_conj @ jump for jump_conj, jump in zip(hconj_jumps, jumps)
        ]
        return H, jumps, hconj_jumps, anticommutator_components


class HamiltonianGenerator(Generator):

    def __init__(self, H: OperatorLike) -> None:
        self.H = H
        self._unitary_cache = {}

    def __add__(self, dynamic: HamiltonianGenerator) -> HamiltonianGenerator:
        if isinstance(dynamic, HamiltonianGenerator):
            if self.dim == dynamic.dim:
                return HamiltonianGenerator(self.H + dynamic.H)
            else:
                raise ValueError(
                    f"Cannot add UnitaryDynamics with dims={self.dim}, {dynamic.dim}"
                )
        return NotImplemented

    @property
    def dim(self) -> int:
        return self.H.dim

    def visitBra(self, psi: Bra, t: Real) -> Bra:
        return 1j * psi @ self.H(t).hConj()

    def visitKet(self, psi: Ket, t: Real) -> Ket:
        return -1j * self.H(t) @ psi

    def visitDensityMatrix(self, rho: DensityMatrix, t: Real) -> DensityMatrix:
        return -1j * self.H(t).commutator(rho)

    def onOperator(self, op: Operator, t: float = 0) -> Operator:
        return 1j * self.H(t).commutator(op)

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> HamiltonianGenerator:
        if not base_dims:
            base_dims = (self.dim,)
        return HamiltonianGenerator(
            self.H.changeHilbertSpace(new_dims, send_to_sites, base_dims)
        )

    def isTimeIndependent(self) -> bool:
        return isinstance(self.H, Operator)

    def unitaryOperator(self, t: Real) -> Operator:
        if t not in self._unitary_cache:
            self._unitary_cache[t] = Operator(expm(-1j * self.H.matrix * t))
        return self._unitary_cache[t]
