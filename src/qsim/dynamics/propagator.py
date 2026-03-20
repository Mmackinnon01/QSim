"""
Time-evolution propagators for quantum dynamics.

This module defines abstract and concrete propagators that integrate
quantum dynamical generators in time.

A propagator is responsible for computing finite-time evolution from
an infinitesimal generator. Different propagators correspond to

- Exact exponentiation (for time-independent Hamiltonians),
- Numerical integration schemes (e.g., Runge–Kutta),
- Or other approximation strategies.

The separation between generator and propagator allows the same
physical model to be evolved using different numerical methods.
"""

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Callable

import numpy as np

from qsim.dynamics.generator import Generator, HamiltonianGenerator
from qsim.ensemble import HilbertSchmidt
from qsim.lin_alg.operator import Operator
from qsim.numeric_solvers.runge_kutta import rungeKutta
from qsim.state.base import QuantumState, StateVisitor
from qsim.state.density_matrix import DensityMatrix
from qsim.state.wave_vector import Bra, Ket


class Propagator(ABC):
    """
    Abstract base class for time-evolution propagators.

    A `Propagator` integrates a `Generator` in time to produce
    finite-time evolution of quantum states or operators.

    Subclasses must implement:

    - `evolve`: evolution of quantum states.
    - `evolveOperator`: evolution of operators.
    - 'callbacks': a list of callables that accept parameters (state, t)
    - 'verbose': settings for level of information output
    """

    def __init__(self, callbacks: list[Callable] = None, verbose: int = 0):
        if callbacks is not None:
            if not isinstance(callbacks, list) or not callable(callbacks[0]):
                raise TypeError(
                    f"Callbacks must be a list of type callable, not {type(callbacks)}"
                )
        self._callbacks = callbacks
        self._verbose = verbose

    @abstractmethod
    def evolve(
        self, gen: Generator, state: QuantumState, t_final: Real, t0: Real = 0
    ) -> QuantumState:
        """
        Evolve a quantum state from time t0 to t_final.

        Parameters
        ----------
        gen : Generator
            Dynamical generator.
        state : QuantumState
            State to evolve.
        t_final : Real
            Target time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        QuantumState
            The evolved state.
        """
        ...

    @abstractmethod
    def evolveOperator(
        self, gen: Generator, op: Operator, t_final: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator from time t0 to t_final, t0>t_final.

        Parameters
        ----------
        gen : Generator
            Dynamical generator.
        op : Operator
            Operator to evolve.
        t_final : Real
            Target time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        Operator
            The evolved operator at t_final = t0-t.
        """
        ...

    def _callback(self, obj: Any, t: Real) -> None:
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback(obj, t)


class ExponentialPropagator(Propagator, StateVisitor):
    """
    Exact propagator for time-independent or discrete time-dependent Hamiltonian dynamics
    The assumption is made that the driving Hamiltonian is constant on the supplied time interval

    This propagator computes evolution using the unitary operator
    U(t) = exp(-i H t), obtained from a `HamiltonianGenerator`.

    It supports evolution of kets, bras, density matrices,
    and operators via visitor-based double dispatch.
    """

    def evolve(
        self,
        gen: HamiltonianGenerator,
        state: QuantumState,
        t_final: Real,
        t0: Real = 0,
    ) -> QuantumState:
        """
        Evolve a quantum state using exact unitary propagation.

        Parameters
        ----------
        gen : HamiltonianGenerator
            Time-independent Hamiltonian generator.
        state : QuantumState
            State to evolve.
        t_final : Real
            Target time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        QuantumState
            The evolved state.
        """
        state = state.accept(self, t_final=t_final, t0=t0, gen=gen)
        self._callback(state, t_final)
        return state

    def visitBra(
        self, psi: Bra, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> Bra:
        U = gen.unitaryOperator(t0, t_final - t0)
        return psi @ U.hConj()

    def visitKet(
        self, psi: Ket, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> Ket:
        U = gen.unitaryOperator(t0, t_final - t0)
        return U @ psi

    def visitDensityMatrix(
        self, rho: DensityMatrix, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> DensityMatrix:
        U = gen.unitaryOperator(t0, t_final - t0)
        return U @ rho @ U.hConj()

    def evolveOperator(
        self, gen: Generator, op: Operator, t_final: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator using exact unitary propagation.

        Parameters
        ----------
        gen : HamiltonianGenerator
            Time-independent Hamiltonian generator.
        op : Operator
            State to evolve.
        t_final : Real
            target time time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        Operator
            The evolved operator at t_final.
        """
        U = gen.unitaryOperator(t0, t0 - t_final)
        self._callback(U, t_final)
        return U.hConj() @ op @ U


class DiagonalPropagator(Propagator, StateVisitor):
    """
    Exact propagator for time-independent diagonalised Hamiltonian dynamics

    This propagator computes evolution by updating the phases of each eigenmode of the Hamiltonian

    It supports evolution of kets, bras, density matrices,
    and operators via visitor-based double dispatch.
    """

    def evolve(
        self,
        gen: HamiltonianGenerator,
        state: QuantumState,
        t_final: Real,
        t0: Real = 0,
    ) -> QuantumState:
        """
        Evolve a quantum state using exact unitary propagation.

        Parameters
        ----------
        gen : HamiltonianGenerator
            Time-independent Hamiltonian generator.
        state : QuantumState
            State to evolve.
        t_final : Real
            Target time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        QuantumState
            The evolved state.
        """
        if not np.all(
            np.abs(gen.H.matrix - np.diag(np.diagonal(gen.H.matrix))) < 10e-12
        ):
            raise ValueError(
                f"Generator Hamiltonian is not diagonal, magnitude of off-diagonal elements is {np.max(np.abs(gen.H.matrix - np.diag(np.diagonal(gen.H.matrix))))} > 10e-12"
            )
        state = state.accept(self, t_final=t_final, t0=t0, gen=gen)
        self._callback(state, t_final)
        return state

    def visitBra(
        self, psi: Bra, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> Bra:
        U = np.diag(gen.unitaryOperator(t0, t_final - t0).matrix)
        return Bra(psi.matrix * U.conj()[None, :])

    def visitKet(
        self, psi: Ket, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> Ket:
        U = np.diag(gen.unitaryOperator(t0, t_final - t0).matrix)
        return Ket(U[:, None] * psi.matrix)

    def visitDensityMatrix(
        self, rho: DensityMatrix, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> DensityMatrix:
        U = np.diag(gen.unitaryOperator(t0, t_final - t0).matrix)
        return DensityMatrix(U[:, None] * rho.matrix * U.conj()[None, :])

    def evolveOperator(
        self, gen: Generator, op: Operator, t_final: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator using exact unitary propagation.

        Parameters
        ----------
        gen : HamiltonianGenerator
            Time-independent Hamiltonian generator.
        op : Operator
            State to evolve.
        t_final : Real
            target time time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        Operator
            The evolved operator at t_final.
        """
        U = np.diag(gen.unitaryOperator(t0, t0 - t_final).matrix)
        self._callback(U, t_final)
        return Operator(U.conj()[:, None] * op.matrix * U[None, :])


class RK4Propagator(Propagator, StateVisitor):
    """
    Fourth-order Runge–Kutta propagator.

    This propagator numerically integrates a general time-dependent
    generator using the classical RK4 scheme with fixed timestep `ts`.

    Suitable for:
    - Time-dependent Hamiltonians
    - GKSL master equations
    - General non-unitary dynamics
    """

    def __init__(
        self,
        ts: Real | None = None,
        callbacks: list[Callable] = None,
        tol: float = 10e-8,
        verbose: int = 0,
    ):
        super().__init__(callbacks=callbacks, verbose=verbose)
        self.ts = ts
        self._ts_cache = {}
        self._tol = tol

    def evolve(
        self, gen: Generator, state: QuantumState, t_final: Real, t0: Real = 0
    ) -> QuantumState:
        """
        Evolve a quantum state using fourth-order Runge–Kutta integration.
                Parameters
        ----------
        gen : Generator
            Infinitesimal generator of the dynamics.
        state : QuantumState
            Initial state.
        t_final : Real
            Evolution duration.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        QuantumState
            The evolved state.
        """
        if self.ts is None:
            ts = self._getAutoTS(gen)
        else:
            ts = self.ts

        t = t0

        while t < t_final:
            if t_final - t < ts:
                timestep = t_final - t
            else:
                timestep = ts

            state = rungeKutta(lambda t_n, s: gen.onState(s, t_n), t, timestep, state)
            t += timestep
            self._callback(state, t)

        return state

    def evolveOperator(
        self, gen: Generator, op: Operator, t_final: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator using fourth-order Runge–Kutta integration.
                Parameters
        ----------
        gen : Generator
            Infinitesimal generator of the dynamics.
        op : Operator
            Initial state.
        t_final : Real
            Evolution duration.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        Operator
            The evolved operator at t_final.
        """
        if self.ts is None:
            ts = self._getAutoTS(gen)
        else:
            ts = self.ts

        t = t0

        while t > t_final:
            if t - t_final < ts:
                timestep = t - t_final
            else:
                timestep = ts

            ## evaluation time of the function is inverted to facilitate backwards evolution while letting timestep be positive
            op = rungeKutta(lambda t_n, s: gen.onOperator(s, t - t_n), 0, timestep, op)
            t -= timestep
            self._callback(op, t)

        return op

    def _getAutoTS(self, gen: Generator) -> float:
        if gen not in self._ts_cache:
            self._ts_cache[gen] = self._identifyTimeStep(gen)
        return self._ts_cache[gen]

    def _identifyTimeStep(self, gen: Generator) -> float:
        ts = 0.01
        t_final = 100 * ts
        inaccurate = True

        while inaccurate:
            if self._verbose == 1:
                print(f"Trialling RK4 propagtor with ts={ts}")
            final_state = self._testStateEvolution(gen, ts, t_final)
            new_state = self._testStateEvolution(gen, ts / 2, t_final)
            if np.linalg.norm(final_state.matrix - new_state.matrix) < self._tol:
                inaccurate = False
                if self._verbose == 1:
                    print(f"RK4 propagtor selected at ts={ts}")
            else:
                ts = ts / 2
                t_final = t_final / 2
                final_state = new_state

        return ts

    def _testStateEvolution(self, gen, ts, t_final):
        state = HilbertSchmidt.generateDM(gen.dim, rng=np.random.default_rng(seed=42))

        t = 0

        while t < t_final:
            if t_final - t < ts:
                timestep = t_final - t
            else:
                timestep = ts

            state = rungeKutta(lambda t_n, s: gen.onState(s, t_n), t, timestep, state)
            t += timestep

        return state
