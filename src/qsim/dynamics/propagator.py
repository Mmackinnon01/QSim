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
        self.solver=rungeKutta

    def evolve(
        self, gen: Generator, state: QuantumState, t_final: Real, t0: Real = 0
    ) -> QuantumState:
        # ... docstrings ...
        
        if self.ts is None:
            ts = self._getAutoTS(gen)
        else:
            ts = self.ts

        state_type = type(state)
        gen_func, input_func = gen.onState(state)
        state_array = state.matrix
        
        # Calculate number of steps
        num_steps = int((t_final - t0) / ts)
        
        if num_steps == 0:
            exact_ts = ts  # Fallback to prevent division by zero
        else:
            exact_ts = (t_final - t0) / num_steps
        
        # 3. Use exact_ts for everything going forward
        time_grid = t0 + np.arange(2 * num_steps + 1) * (exact_ts / 2.0)
        
        # input_func now receives a proper 1D numpy array
        input_arrays = input_func(time_grid)

        dim = state_array.shape
        trajectory = np.empty((num_steps + 1, *dim), dtype=np.complex128)

        self.solver(gen_func, state_array, exact_ts, input_arrays, trajectory)

        if self._callbacks is not None:
            for i, step_state in enumerate(trajectory): # FIX 4: Rename to avoid shadowing
                current_time = t0 + i * ts              # FIX 4: Include t0
                print(t0, i, ts)
                self._callback(state_type(step_state), current_time)

        return state_type(trajectory[-1])

    def evolveOperator(
        self, gen: Generator, op: Operator, t_final: Real, t0: Real = 0
    ) -> Operator:
        """
        Evolve a quantum operator using fourth-order Runge–Kutta integration.
        # ... docstrings ...
        """
        if self.ts is None:
            ts = self._getAutoTS(gen)
        else:
            ts = self.ts

        # IMPORTANT: Ensure gen.onOperator is updated to return BOTH 
        # gen_func and input_func, exactly like gen.onState does.
        gen_func, input_func = gen.onOperator(op)
        
        op_array = op.matrix

        # 1. Handle time direction robustly (Heisenberg picture usually goes backwards)
        # This allows t0 > t_final OR t_final > t0 safely.
        time_diff = t_final - t0
        direction = 1 if time_diff >= 0 else -1
        
        # 2. Calculate integer steps 
        num_steps = int(round(abs(time_diff) / ts))
        
        # 3. Generate the exact time grid, stepping in the correct direction
        # Shape: (2 * num_steps + 1,) for t, t+dt/2, and t+dt evaluations
        if num_steps == 0:
            exact_ts = ts  # Fallback to prevent division by zero
        else:
            exact_ts = (t0 - t_final) / num_steps
        
        # 3. Use exact_ts for everything going forward
        time_grid = t0 + direction * np.arange(2 * num_steps + 1) * (exact_ts / 2.0)
        
        # 4. Pre-evaluate all time-dependent operator matrices in C
        input_arrays = input_func(time_grid)
     
        # 5. Execute the monolithic C-compiled solver
        # Note: Your JIT solver must be updated to loop over `num_steps` 
        # and return the whole trajectory, exactly like the state solver.
        dim = op_array.shape
        trajectory = np.empty((num_steps + 1, *dim), dtype=np.complex128)
        self.solver(
            gen_func, op_array, exact_ts, input_arrays, trajectory
        )

        # 6. Process callbacks using the exact trajectory times
        if self._callbacks is not None:
            for i, step_op in enumerate(trajectory):
                current_time = t0 + direction * i * ts
                self._callback(Operator(step_op), current_time)

        return Operator(trajectory[-1])

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

        state_type = type(state)
        gen_func, input_func = gen.onState(state)
        state_array = state.matrix
        
        # Calculate number of steps
        num_steps = int((t_final) / ts)
        
        # FIX 1, 2, & 3: Use NumPy, include t0, and add +1 for the final k4 step
        time_grid = 0 + np.arange(2 * num_steps + 1) * (ts / 2.0)
        
        # input_func now receives a proper 1D numpy array
        input_arrays = input_func(time_grid)

        dim = state_array.shape
        trajectory = np.empty((num_steps + 1, *dim), dtype=np.complex128)
        self.solver(gen_func, state_array, ts, input_arrays, trajectory)
        
        return state_type(trajectory[-1])

