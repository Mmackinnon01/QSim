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

    def __init__(self, ts: Real = 0.001, callbacks: list[Callable] = None):
        self.ts = ts
        if callbacks is not None:
            if not isinstance(callbacks, list) or not callable(callbacks[0]):
                raise TypeError(
                    f"Callbacks must be a list of type callable, not {type(callbacks)}"
                )
        self._callbacks = callbacks

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
        if not gen.isTimeIndependent():
            raise ValueError(
                "Exponential propagation only valid for time-independent dynamics"
            )
        state = state.accept(self, t_final=t_final, t0=t0, gen=gen)
        self._callback(state, t_final)
        return state

    def visitBra(
        self, psi: Bra, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> Bra:
        U = gen.unitaryOperator(t_final - t0)
        return psi @ U.hConj()

    def visitKet(
        self, psi: Ket, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> Ket:
        U = gen.unitaryOperator(t_final - t0)
        return U @ psi

    def visitDensityMatrix(
        self, rho: DensityMatrix, gen: HamiltonianGenerator, t_final: Real, t0: Real = 0
    ) -> DensityMatrix:
        U = gen.unitaryOperator(t_final - t0)
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
        U = gen.unitaryOperator(t0 - t_final)
        self._callback(U, t_final)
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
        t = t0

        while t < t_final:
            if t_final - t < self.ts:
                timestep = t_final - t
            else:
                timestep = self.ts

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
        t = t0

        while t > t_final:
            if t - t_final < self.ts:
                timestep = t - t_final
            else:
                timestep = self.ts

            ## evaluation time of the function is inverted to facilitate backwards evolution while letting timestep be positive
            op = rungeKutta(lambda t_n, s: gen.onOperator(s, t - t_n), 0, timestep, op)
            t -= timestep
            self._callback(op, t)

        return op
