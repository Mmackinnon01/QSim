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
from sre_parse import State
from typing import Any, Callable

from qsim.dynamics.generator import Generator, HamiltonianGenerator
from qsim.numeric_solvers.runge_kutta import rungeKutta
from qsim.operator.base import Operator
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
    """

    def __init__(self, ts=0.001, callbacks: list[Callable] = None):
        self.ts = ts
        if callbacks is not None:
            if not isinstance(callbacks, list) or not callable(callbacks[0]):
                raise TypeError(
                    f"Callbacks must be a list of type callable, not {type(callbacks)}"
                )
        self._callbacks = callbacks

    @abstractmethod
    def evolve(
        self, gen: Generator, state: QuantumState, t: Real, t0: Real = 0
    ) -> QuantumState:
        """
        Evolve a quantum state from time t0 to t0 + t.

        Parameters
        ----------
        gen : Generator
            Dynamical generator.
        state : QuantumState
            State to evolve.
        t : Real
            Evolution time.
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
        self, gen: Generator, operator: Operator, t: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator from time t0 to t0 - t.

        Parameters
        ----------
        gen : Generator
            Dynamical generator.
        op : Operator
            Operator to evolve.
        t : Real
            Evolution time.
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
    Exact propagator for time-independent Hamiltonian dynamics.

    This propagator computes evolution using the unitary operator
    U(t) = exp(-i H t), obtained from a time-independent
    `HamiltonianGenerator`.

    It supports evolution of kets, bras, density matrices,
    and operators via visitor-based double dispatch.
    """

    def evolve(
        self,
        gen: HamiltonianGenerator,
        state: QuantumState,
        t: Real,
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
        t : Real
            Evolution time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        QuantumState
            The evolved state.
        """
        if not gen.isTimeIndependent():
            raise ValueError(
                "Exponential propagation only valid for time-independent dynamics"
            )
        state = state.accept(self, t=t, t0=t0, gen=gen)
        self._callback(state, t0 + t)
        return state

    def visitBra(
        self, psi: Bra, gen: HamiltonianGenerator, t: Real, t0: Real = 0
    ) -> Bra:
        U = gen.unitaryOperator(t)
        return psi @ U.hConj()

    def visitKet(
        self, psi: Ket, gen: HamiltonianGenerator, t: Real, t0: Real = 0
    ) -> Ket:
        U = gen.unitaryOperator(t)
        return U @ psi

    def visitDensityMatrix(
        self, rho: DensityMatrix, gen: HamiltonianGenerator, t: Real, t0: Real = 0
    ) -> DensityMatrix:
        U = gen.unitaryOperator(t)
        return U @ rho @ U.hConj()

    def evolveOperator(
        self, gen: Generator, op: Operator, t: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator using exact unitary propagation.

        Parameters
        ----------
        gen : HamiltonianGenerator
            Time-independent Hamiltonian generator.
        op : Operator
            State to evolve.
        t : Real
            Evolution time.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        Operator
            The evolved operator at t_final = t-t0.
        """
        U = gen.unitaryOperator(t)
        self._callback(U, t0 - t)
        return U.hConj() @ op @ U


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

    def evolve(
        self, gen: Generator, state: QuantumState, t: Real, t0: Real = 0
    ) -> QuantumState:
        """
        Evolve a quantum state using fourth-order Runge–Kutta integration.
                Parameters
        ----------
        gen : Generator
            Infinitesimal generator of the dynamics.
        state : QuantumState
            Initial state.
        t : Real
            Evolution duration.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        QuantumState
            The evolved state.
        """
        t_evolve = t0
        t_final = t0 + t
        while t_evolve < t_final:
            if t_final - t_evolve < self.ts:
                timestep = t_final - t_evolve
            else:
                timestep = self.ts

            state = rungeKutta(
                lambda state: gen.onState(state, t_evolve), timestep, state
            )
            t_evolve += timestep
            self._callback(state, t_evolve)

        return state

    def evolveOperator(
        self, gen: Generator, op: Operator, t: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator using fourth-order Runge–Kutta integration.
                Parameters
        ----------
        gen : Generator
            Infinitesimal generator of the dynamics.
        op : Operator
            Initial state.
        t : Real
            Evolution duration.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        Operator
            The evolved operator at t_final = t0 - t.
        """
        t_evolve = t0
        t_final = t0 - t
        while t_evolve > t_final:
            if t_evolve - t_final < self.ts:
                timestep = t_evolve - t_final
            else:
                timestep = self.ts

            op = rungeKutta(lambda op: gen.onOperator(op, t_evolve), timestep, op)
            t_evolve -= timestep
            self._callback(op, t_evolve)
        return op
