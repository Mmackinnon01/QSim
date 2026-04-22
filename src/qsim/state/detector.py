from abc import ABC, abstractmethod
from functools import reduce
from numbers import Real

import numpy as np
from numpy.random import f

from qsim.lin_alg.observable import Observable
from qsim.lin_alg.operator import I, Operator
from qsim.state.density_matrix import DensityMatrix
from qsim.state.wave_vector import Bra, Ket

from .base import QuantumState, StateVisitor


class DetectorInterface(ABC, StateVisitor):

    _dims: tuple
    _targs: tuple

    def detect(self, state: QuantumState, shots: int = -1):
        return state.accept(self, shots=shots)

    def visitBra(self, psi: Bra, shots: int = -1) -> Real:
        return self.visitKet(psi.hConj(), shots)

    def visitKet(self, psi: Ket, shots: int = -1) -> Real:
        if len(self._targs) < len(self._dims):
            rho = psi.partialTrace(self._dims, self._targs)
            if shots == -1:
                return self._infiniteStatisticsDensity(rho)
            else:
                return self._finiteStatisticsDensity(rho, shots)

        if shots == -1:
            return self._infiniteStatisticsKet(psi)
        else:
            return self._finiteStatisticsKet(psi, shots)

    def visitDensityMatrix(self, rho: DensityMatrix, shots: int = -1) -> Real:
        if len(self._targs) < len(self._dims):
            rho = rho.partialTrace(self._dims, self._targs)
        if shots == -1:
            return self._infiniteStatisticsDensity(rho)
        else:
            return self._finiteStatisticsDensity(rho, shots)

    @abstractmethod
    def _infiniteStatisticsDensity(self, rho: DensityMatrix) -> Real: ...

    @abstractmethod
    def _infiniteStatisticsKet(self, psi: Ket) -> Real: ...

    @abstractmethod
    def _finiteStatisticsDensity(self, rho: DensityMatrix, shots: int) -> Real: ...

    @abstractmethod
    def _finiteStatisticsKet(self, psi: Ket, shots: int) -> Real: ...

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


class ObservableDetector(DetectorInterface):

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

    def _infiniteStatisticsDensity(self, rho: DensityMatrix) -> Real:
        return (self._ob @ rho).trace().real

    def _infiniteStatisticsKet(self, psi: Ket) -> Real:
        return (psi.hConj() @ self._ob @ psi).real

    def _finiteStatisticsDensity(self, rho: DensityMatrix, shots: int) -> Real:
        eigvals, eigvecs = self._ob.eigenvalues, self._ob.eigenvectors
        probs = np.real(np.einsum("ij,ji->i", eigvecs.conj().T @ rho.matrix, eigvecs))
        return self._finiteStatistics(probs, eigvals, shots)

    def _finiteStatisticsKet(self, psi: Ket, shots: int) -> Real:
        eigvals, eigvecs = self._ob.eigenvalues, self._ob.eigenvectors
        amps = eigvecs.conj().T @ psi.matrix
        probs = np.abs(amps.squeeze()) ** 2
        return self._finiteStatistics(probs, eigvals, shots)


class POVMDetector(DetectorInterface):
    def __init__(
        self,
        povm: list[Operator],
        outcomes: list[Real],
        dims: tuple[int, ...] | None = None,
        target_sites: tuple[int, ...] | None = None,
    ) -> None:
        if not dims:
            dims = (povm[0].dim,)
        if not target_sites:
            target_sites = tuple(range(len(dims)))
        if reduce(lambda x, y: x + y, povm) != I(povm[0].dim):
            raise ValueError("Invalid POVM, does not sum to identity")
        if np.any([not op.isSemiPositive() for op in povm]):
            raise ValueError("Invalid POVM, contains negative operators")

        self._povm = povm
        self._outcomes = outcomes
        self._dims = dims
        self._targs = target_sites

    def _infiniteStatisticsDensity(self, rho: DensityMatrix) -> Real:
        return sum(
            [
                (op @ rho).trace().real * outcome
                for op, outcome in zip(self._povm, self._outcomes)
            ]
        )

    def _infiniteStatisticsKet(self, psi: Ket) -> Real:
        return sum(
            [
                (psi.hConj() @ op @ psi).real * outcome
                for op, outcome in zip(self._povm, self._outcomes)
            ]
        )

    def _finiteStatisticsDensity(self, rho: DensityMatrix, shots: int) -> Real:
        return self._finiteStatistics(
            [(op @ rho).trace().real for op in self._povm], self._outcomes, shots
        )

    def _finiteStatisticsKet(self, psi: Ket, shots: int) -> Real:
        return self._finiteStatistics(
            [(psi.hConj() @ op @ psi).real for op in self._povm], self._outcomes, shots
        )
