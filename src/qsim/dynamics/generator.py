from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from numbers import Real
from typing import Self

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.linalg import eig, expm, inv, svd

from qsim.lin_alg import I, Operator, Vector
from qsim.lin_alg.operator import OperatorLike
from qsim.lin_alg.transforms import unvectorise, vectorise
from qsim.state import Bra, DensityMatrix, Ket, QuantumState, StateVisitor


@nb.njit(fastmath=True)
def _fast_lindblad_heisenberg(target_arr: np.ndarray, op_arr: np.ndarray, out: np.ndarray):
    H_arr = op_arr[0]
    jumps_arr = op_arr[1:]
    
    # 1. Overwrite the pre-allocated 'out' array instead of creating a new one
    out[:] = 1j * (H_arr @ target_arr - target_arr @ H_arr)
    
    for i in range(len(jumps_arr)):
        L = jumps_arr[i]
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        
        # 2. Add to 'out' in-place
        out += (L_dag @ target_arr @ L) - 0.5 * (L_dag_L @ target_arr + target_arr @ L_dag_L)

@nb.njit(fastmath=True)
def _fast_lindblad_schrodinger(rho_arr: np.ndarray, op_arr: np.ndarray, out: np.ndarray):
    H_arr = op_arr[0]
    jumps_arr = op_arr[1:]
    
    # 1. Overwrite the pre-allocated 'out' array
    out[:] = -1j * (H_arr @ rho_arr - rho_arr @ H_arr)
    
    for i in range(len(jumps_arr)):
        L = jumps_arr[i]
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        
        # 2. Add to 'out' in-place
        out += (L @ rho_arr @ L_dag) - 0.5 * (L_dag_L @ rho_arr + rho_arr @ L_dag_L)


class Generator(ABC, StateVisitor):

    def __init__(self):
        self._compiled_fns = {}

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
        super().__init__()
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
        backend = _fast_lindblad_schrodinger if is_density_matrix else _fast_lindblad_heisenberg
        H_t = self.H.compile()
        jumps = [jump.compile() for jump in self.jumps]

        def inputs_func(ts):
            # 1. Evaluate the Hamiltonian to get the shape and data
            H_eval = H_t(ts)
            T, N, _ = H_eval.shape  # T is time steps, N is matrix dimension
            K = 1 + len(jumps)      # Total number of operators

            # 2. Allocate the final memory block ONCE
            # (Using axis=1 means shape should be (T, K, N, N) so op_arr[i] gives operators for step i)
            op_arrays = np.empty((T, K, N, N), dtype=np.complex128)

            # 3. Fill the slots directly
            op_arrays[:, 0] = H_eval
            for i, jump in enumerate(jumps):
                op_arrays[:, i + 1] = jump(ts)
            return op_arrays
            
        return backend, inputs_func

    def onOperator(self, op) -> callable:
        if 'op' not in self._compiled_fns.keys():
            self._compiled_fns['op'] = self._build_generator(is_density_matrix=False)
        return self._compiled_fns['op']

    def visitDensityMatrix(self, rho) -> callable:
        if 'dm' not in self._compiled_fns.keys():
            self._compiled_fns['dm'] = self._build_generator(is_density_matrix=True)
        return self._compiled_fns['dm']

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
    

@nb.njit(fastmath=True)
def hamiltonian_bra(psi_arr: np.ndarray, op_arr: np.ndarray, out: np.ndarray)->np.ndarray:
    out[:] = 1j * psi_arr @ op_arr

@nb.njit(fastmath=True)
def hamiltonian_ket(psi_arr: np.ndarray, op_arr: np.ndarray, out: np.ndarray)->np.ndarray:
    out[:] = -1j * op_arr @ psi_arr

@nb.njit(fastmath=True)
def hamiltonian_dm(psi_arr: np.ndarray, op_arr: np.ndarray, out: np.ndarray)->np.ndarray:
    out[:] = -1j * (op_arr @ psi_arr - psi_arr @ op_arr)

@nb.njit(fastmath=True)
def hamiltonian_op(psi_arr: np.ndarray, op_arr: np.ndarray, out: np.ndarray)->np.ndarray:
    out[:] = 1j * (op_arr @ psi_arr - psi_arr @ op_arr)


class HamiltonianGenerator(Generator):

    def __init__(self, H: OperatorLike) -> None:
        super().__init__()
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
        if 'bra' not in self._compiled_fns.keys():
            H_t = self.H.compile()
            def inputs_func(ts):
                return H_t(ts)

            self._compiled_fns['bra'] = (hamiltonian_bra, inputs_func)
        return self._compiled_fns['bra']

    def visitKet(self, psi: Ket) -> Ket:
        if 'ket' not in self._compiled_fns.keys():
            H_t = self.H.compile()
            def inputs_func(ts):
                return H_t(ts)

            self._compiled_fns['ket'] =  (hamiltonian_ket, inputs_func)
        return self._compiled_fns['ket']

    def visitDensityMatrix(self, rho: DensityMatrix) -> DensityMatrix:
        if 'dm' not in self._compiled_fns.keys():
            H_t = self.H.compile()
            def inputs_func(ts):
                return H_t(ts)

            self._compiled_fns['dm'] = (hamiltonian_dm, inputs_func)
        return self._compiled_fns['dm']

    def onOperator(self, op: Operator) -> Operator:
        if 'op' not in self._compiled_fns.keys():
            H_t = self.H.compile()
            def inputs_func(ts):
                return H_t(ts)

            self._compiled_fns['op'] = (hamiltonian_op, inputs_func)
        return self._compiled_fns['op']

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

@nb.njit(fastmath=True)
def liouvillian_generator(psi_arr: np.ndarray, op_arr: np.ndarray, out: np.ndarray)->np.ndarray:
    out[:] = op_arr @ psi_arr


class LiouvillianGenerator(Generator):

    def __init__(self, L: OperatorLike) -> None:
        super().__init__()
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
        if 'ket' not in self._compiled_fns.keys():
            L_t = self.L.compile()
            def inputs_func(ts):
                return L_t(ts)

            self._compiled_fns['ket'] =  (liouvillian_generator, inputs_func)
        return self._compiled_fns['ket']

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

    def steadyState(self, t: Real = 0, rho0: DensityMatrix | None = None):
        # 1. Check cache for the left and right biorthonormal bases
        if t not in self._spectral_cache:
            L_mat = self.L(t).matrix
            
            # SVD is highly optimized and much faster than eig
            U, s, Vh = svd(L_mat)
            
            # Find the indices of the zero singular values (the steady states)
            tol = 10e-12
            null_idx = np.where(s < tol)[0]
            
            if len(null_idx) == 0:
                raise ValueError("No steady state found; Liouvillian is non-singular.")

            # 2. Extract Null Spaces
            # Right null vectors (columns of V)
            R = Vh[null_idx, :].conj().T
            
            # Left null vectors (columns of U)
            L_left = U[:, null_idx]
            
            # 3. Fast Biorthonormalization using pure linear algebra
            # We need L_left^H @ R = Identity. 
            M = L_left.conj().T @ R
            L_biorth = L_left @ inv(M.conj().T)
            
            # Cache the bases, not the final state, because the final state depends on rho0
            self._spectral_cache[t] = (R, L_biorth)
            
        # Retrieve the bases from cache
        # Retrieve the bases from cache
        R, L_biorth = self._spectral_cache[t]
        
        # 4. Project the initial state
        # Extract the raw NumPy array from your custom Vector object and flatten it to 1D
        rho0_vec = vectorise(rho0).matrix.flatten()
        
        # Now NumPy can execute the fast C-level matrix-vector product
        steady_vec = R @ (L_biorth.conj().T @ rho0_vec)
        
        # 5. Enforce Trace = 1 to clean up numerical noise
        # Wrap the raw NumPy array back into your Vector class so unvectorise() recognizes it
        steady_obj = unvectorise(Vector(steady_vec))
        
        # Extract the 2D matrix for the trace calculation
        state_matrix = steady_obj.matrix
        
        trace_val = state_matrix.trace()
        if abs(trace_val) > 10e-10:
            state_matrix = state_matrix / trace_val
            
        return DensityMatrix(state_matrix)

    def plotSpectrum(self, t: Real = 0, ax: plt.axes | None = None) -> plt.axes:
        eigs, lv, rv = self.spectralDecomposition(t)
        if not ax:
            ax = plt.subplot()
        ax.scatter(np.real(eigs), np.imag(eigs))
        ax.set_xlabel(r"Re$(\lambda_i)$")
        ax.set_ylabel(r"Im$(\lambda_i)$")
        return ax


