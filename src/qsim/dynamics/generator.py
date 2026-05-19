from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from numbers import Real
from typing import Self

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.linalg import eig, expm

from qsim.lin_alg import I, Operator, Vector
from qsim.lin_alg.operator import OperatorLike
from qsim.lin_alg.transforms import unvectorise, vectorise
from qsim.state import Bra, DensityMatrix, Ket, QuantumState, StateVisitor


@nb.njit
def _fast_lindblad_heisenberg(op_arr: np.ndarray, H_arr: np.ndarray, jumps_arr: np.ndarray) -> np.ndarray:
    out = 1j * (H_arr @ op_arr - op_arr @ H_arr)  # +1j
    for i in range(len(jumps_arr)):
        L = jumps_arr[i]
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        
        # L_dag @ A @ L
        out += (L_dag @ op_arr @ L) - 0.5 * (L_dag_L @ op_arr + op_arr @ L_dag_L)
    return out

@nb.njit
def _fast_lindblad_schrodinger(rho_arr: np.ndarray, H_arr: np.ndarray, jumps_arr: np.ndarray) -> np.ndarray:
    out = -1j * (H_arr @ rho_arr - rho_arr @ H_arr) # -1j
    for i in range(len(jumps_arr)):
        L = jumps_arr[i]
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        
        # L @ rho @ L_dag
        out += (L @ rho_arr @ L_dag) - 0.5 * (L_dag_L @ rho_arr + rho_arr @ L_dag_L)
    return out


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

    def onState(self, state: QuantumState) -> QuantumState:
        return state.accept(self)

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


    def _build_generator(self, is_density_matrix: bool):
        """Shared setup logic to avoid code duplication."""
        get_H = self.H.compile()
        get_jumps = [jump.compile() for jump in self.jumps]
        
        # Pick the correct machine-code backend just ONCE during setup
        backend = _fast_lindblad_schrodinger if is_density_matrix else _fast_lindblad_heisenberg

        def generator(arr: np.ndarray, t: float = 0.0) -> np.ndarray:
            # Bulletproof type checking (fixes the BLAS integer crash)
            if arr.dtype != np.complex128:
                arr = arr.astype(np.complex128)
                
            H_t = get_H(t)
            
            if len(get_jumps) > 0:
                jumps_t = np.array([j(t) for j in get_jumps], dtype=np.complex128)
            else:
                jumps_t = np.zeros((0, H_t.shape[0], H_t.shape[1]), dtype=np.complex128)
                
            # Call whichever backend was selected during setup
            return backend(arr, H_t, jumps_t)
            
        return generator


    def onOperator(self, op) -> callable:
        return self._build_generator(is_density_matrix=False)

    def visitDensityMatrix(self, rho) -> callable:
        return self._build_generator(is_density_matrix=True)

    def visitBra(self, psi: Bra) -> TypeError:
        raise TypeError("GKSL master equation not valid for wavevector input")

    def visitKet(self, psi: Ket) -> TypeError:
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

    def visitBra(self, psi: Bra) -> Bra:
        H_t = self.H.compile()

        def f(psi: np.ndarray, t: float = 0)->np.ndarray:
            return 1j * psi @ H_t(t)
        return f


    def visitKet(self, psi: Ket) -> Ket:
        H_t = self.H.compile()

        def f(psi: np.ndarray, t: float = 0)->np.ndarray:
            return -1j * H_t(t) @ psi
        return f

    def visitDensityMatrix(self, rho: DensityMatrix) -> DensityMatrix:
        H_t = self.H.compile()

        def f(rho: np.ndarray, t: float = 0)->np.ndarray:
            H_eval = H_t(t)
            return -1j * (H_eval @ rho - rho @ H_eval)
        return f

    def onOperator(self, op: Operator) -> Operator:
        H_t = self.H.compile()

        def f(op: np.ndarray, t: float = 0)->np.ndarray:
            H_eval = H_t(t)
            return 1j * (H_eval @ op - op @ H_eval)
        return f

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

    def visitBra(self, psi: Bra) -> Bra:
        raise TypeError("Superoperator generator does not work on Bra")

    def visitKet(self, psi: Ket) -> Ket:
        L_t = self.L.compile()

        def f(psi: np.ndarray, t: float = 0)->np.ndarray:
            return L_t(t) @ psi
        return f

    def visitDensityMatrix(self, rho: DensityMatrix) -> DensityMatrix:
        raise TypeError("Superoperator generator does not work on Density Matrices")

    def onOperator(self, op: Operator) -> Operator:
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
