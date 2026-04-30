from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from multiprocessing import Value
from numbers import Real
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig, expm

from qsim.lin_alg import I, Operator, Vector
from qsim.lin_alg.operator import OperatorLike
from qsim.lin_alg.transforms import unvectorise, vectorise
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
            self.anticommutator_components = [
                jump.hConj() @ jump for jump in self.jumps
            ]
        else:
            self.jumps = []
            self.anticommutator_components = []

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
        if not jumps:
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
        if not jumps:
            return unitary_component
        dissipative_component = reduce(
            lambda x, y: x + y,
            [
                L @ rho @ L_dag - 0.5 * (L2 @ rho + rho @ L2)
                for L, L_dag, L2 in zip(jumps, hJumps, ac_components)
            ],
        )
        return unitary_component + dissipative_component

    def visitBra(self, psi: Bra, t: float) -> TypeError:
        raise TypeError("GKSL master equation not valid for wavevector input")

    def visitKet(self, psi: Ket, t: float) -> TypeError:
        raise TypeError("GKSL master equation not valid for wavevector input")

    def _evaluateOperators(self, t: Real):
        H = self.H(t)
        jumps = [jump(t) for jump in self.jumps]
        hconj_jumps = [jump.hConj() for jump in jumps]
        anticommutator_components = [c(t) for c in self.anticommutator_components]
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

    def unitaryOperator(self, t: Real, delta_t: Real) -> Operator:
        # rounding to avoid machine precision errors in computing delta t resulting in multiple expm
        delta_t = np.round(delta_t, 14)
        if delta_t not in self._unitary_cache:
            self._unitary_cache[delta_t] = Operator(
                expm(-1j * self.H(t).matrix * delta_t)
            )
        return self._unitary_cache[delta_t]


class LiouvillianGenerator(Generator):

    def __init__(self, L: OperatorLike) -> None:
        self.L = L
        self._exponential_cache = {}
        self._spectral_cache = {}

    def __add__(self, dynamic: LiouvillianGenerator) -> LiouvillianGenerator:
        if isinstance(dynamic, HamiltonianGenerator):
            if self.dim == dynamic.dim:
                return LiouvillianGenerator(self.L + dynamic.L)
            else:
                raise ValueError(
                    f"Cannot add LiouvillianGenerators with dims={self.dim}, {dynamic.dim}"
                )
        return NotImplemented

    @classmethod
    def fromGKSL(cls, gen: GKSLGenerator) -> LiouvillianGenerator:
        dim = gen.H.dim
        hermitian_component = -1j * ((I(dim) ^ gen.H) - (gen.H.T ^ I(dim)))
        if gen.jumps:
            non_hermitian_component = reduce(
                lambda x, y: x + y,
                [
                    (jump.conj() ^ jump)
                    - 0.5
                    * (
                        (I(dim) ^ (jump.hConj() @ jump))
                        + ((jump.hConj() @ jump).T ^ I(dim))
                    )
                    for jump in gen.jumps
                ],
            )
            return LiouvillianGenerator(hermitian_component + non_hermitian_component)
        else:
            return LiouvillianGenerator(hermitian_component)

    @property
    def dim(self) -> int:
        return self.L.dim

    def visitBra(self, psi: Bra, t: Real) -> Bra:
        raise TypeError("Superoperator generator does not work on Bra")

    def visitKet(self, psi: Ket, t: Real) -> Ket:
        return self.L(t) @ psi

    def visitDensityMatrix(self, rho: DensityMatrix, t: Real) -> DensityMatrix:
        raise TypeError("Superoperator generator does not work on Density Matrices")

    def onOperator(self, op: Operator, t: float = 0) -> Operator:
        raise TypeError("Superoperator generator does not work on Operators")

    def changeHilbertSpace(
        self,
        new_dims: tuple[int, ...],
        send_to_sites: tuple[int, ...],
        base_dims: tuple[int, ...] | None = None,
    ) -> LiouvillianGenerator:
        if not base_dims:
            base_dims = (self.dim,)
        return LiouvillianGenerator(
            self.L.changeHilbertSpace(new_dims, send_to_sites, base_dims)
        )

    def unitaryOperator(self, t: Real, delta_t: Real) -> Operator:
        if t not in self._exponential_cache:
            self._exponential_cache[t] = Operator(expm(self.L(t).matrix * delta_t))
        return self._exponential_cache[t]

    def spectralDecomposition(
        self, t: Real = 0, biorthonomalise: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if t not in self._spectral_cache:
            eigs, lv, rv = eig(self.L(t).matrix, left=True)
            lv = [Vector(l.conj().reshape(1, -1)) for l in lv.T]
            rv = [Vector(r) for r in rv.T]
            eigs, lv, rv = map(
                np.array, zip(*sorted(zip(eigs, lv, rv), key=lambda x: -x[0]))
            )
            if biorthonomalise:
                for i, cond in enumerate(np.isclose(eigs, 0, rtol=10e-10)):
                    if cond:
                        trace = unvectorise(rv[i]).trace()
                        if trace > 10e-10:
                            rv[i] = rv[i] / trace
                lv = [left / (left @ right) for left, right in zip(lv, rv)]
            self._spectral_cache[t] = (eigs, lv, rv)
        return self._spectral_cache[t]

    def plotSpectrum(self, t: Real = 0, ax: plt.axes | None = None) -> plt.axes:
        eigs, lv, rv = self.spectralDecomposition(t)
        if not ax:
            ax = plt.subplot()
        ax.scatter(np.real(eigs), np.imag(eigs))
        ax.set_xlabel(r"Re$(\lambda_i)$")
        ax.set_ylabel(r"Im$(\lambda_i)$")
        return ax

    def steadyState(self, t: Real = 0, rho0: DensityMatrix | None = None):
        eigs, lv, rv = self.spectralDecomposition(t, biorthonomalise=True)

        if sum(np.abs(eigs) < 10e-15) > 1:
            return DensityMatrix(sum(
                [
                    l @ vectorise(rho0) * unvectorise(r)
                    for eig, l, r in zip(eigs, lv, rv)
                    if np.abs(eig) < 10e-15
                ]
            ))
        else:
            return DensityMatrix(unvectorise(rv[0]))
