from numbers import Real

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

    def detect(self, state: QuantumState):
        return state.accept(self)

    def visitBra(self, psi: Bra) -> Real:
        if len(self._targs) < len(self._dims):
            rho = psi.partialTrace(self._dims, self._targs)
            return (self._ob @ rho).trace()
        return (psi @ self._ob @ psi.hConj()).real

    def visitKet(self, psi: Ket) -> Real:
        if len(self._targs) < len(self._dims):
            rho = psi.partialTrace(self._dims, self._targs)
            return (self._ob @ rho).trace()
        return (psi.hConj() @ self._ob @ psi).real

    def visitDensityMatrix(self, rho: DensityMatrix) -> Real:
        if len(self._targs) < len(self._dims):
            rho = rho.partialTrace(self._dims, self._targs)
        return (self._ob @ rho).trace().real
