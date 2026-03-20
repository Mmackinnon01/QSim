from numbers import Real

import numpy as np

from qsim.lin_alg.observable import Observable
from qsim.lin_alg.operator import Operator
from qsim.state.density_matrix import DensityMatrix
from qsim.state.wave_vector import Bra, Ket

from .base import QuantumState, StateVisitor


class Detector(StateVisitor):

    def __init__(
        self,
        observable: Observable | Operator,
        dims: tuple[int, ...] | None = None,
        target_sites: tuple[int, ...] | None = None,
    ) -> None:
        if isinstance(observable, Operator):
            observable = Observable(observable.matrix)
        if not dims:
            dims = (observable.dim,)
        if not target_sites:
            target_sites = tuple(range(len(dims)))

        self._ob = observable
        self._dims = dims
        self._targs = target_sites

    def detect(self, state: QuantumState, shots: int = -1):
        return state.accept(self, shots=shots)

    def visitBra(self, psi: Bra, shots: int = -1) -> Real:
        return self.visitKet(psi.hConj(), shots)

    def visitKet(self, psi: Ket, shots: int = -1) -> Real:
        if len(self._targs) < len(self._dims):
            rho = psi.partialTrace(self._dims, self._targs)
            if shots == -1:
                return (self._ob @ rho).trace()
            else:
                return self._finiteStatisticsDensity(rho, shots)

        if shots == -1:
            return (psi.hConj() @ self._ob @ psi).real
        else:
            return self._finiteStatisticsKet(psi, shots)

    def visitDensityMatrix(self, rho: DensityMatrix, shots: int = -1) -> Real:
        if len(self._targs) < len(self._dims):
            rho = rho.partialTrace(self._dims, self._targs)
        if shots == -1:
            return (self._ob @ rho).trace().real
        else:
            return self._finiteStatisticsDensity(rho, shots)

    def _finiteStatisticsDensity(self, rho: DensityMatrix, shots: int) -> Real:
        eigvals, eigvecs = self._ob.eigenvalues, self._ob.eigenvectors
        probs = np.real(np.einsum("ij,ji->i", eigvecs.conj().T @ rho.matrix, eigvecs))
        return self._finiteStatistics(probs, eigvals, shots)

    def _finiteStatisticsKet(self, psi: Ket, shots: int) -> Real:
        eigvals, eigvecs = self._ob.eigenvalues, self._ob.eigenvectors
        amps = eigvecs.conj().T @ psi.matrix
        probs = np.abs(amps.squeeze()) ** 2
        return self._finiteStatistics(probs, eigvals, shots)

    def _finiteStatistics(
        self, probs: np.ndarray, eigvals: np.ndarray, shots: int
    ) -> Real:
        if not np.isclose(sum(probs), 1):
            raise ValueError("The probabilities must sum to 1.")
        return np.real(
            np.sum(
                [
                    eigvals[i] * val
                    for i, val in enumerate(np.random.multinomial(shots, probs))
                ]
            )
            / shots
        )
