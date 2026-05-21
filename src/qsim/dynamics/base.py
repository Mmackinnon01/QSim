"""
Base abstraction for quantum dynamics.

This module defines the `Dynamics` class, which composes a
`Generator` (the infinitesimal generator of the dynamics)
and a `Propagator` (the integration / time-evolution strategy).

The separation between generator and propagator allows:
- Reuse of the same generator with different integration schemes
  (e.g., exact exponentiation, RK4, Trotter, etc.).
- Support for both time-dependent and time-independent dynamics.
- Application to both quantum states and operators.
"""

from numbers import Real
from typing import Any, Callable, Iterable

from tqdm.auto import tqdm

from qsim.dynamics.generator import Generator
from qsim.dynamics.propagator import Propagator
from qsim.lin_alg import Operator
from qsim.lin_alg.operator import Operator
from qsim.state import QuantumState
from qsim.state.base import QuantumState


class Dynamics:
    """
    High-level wrapper representing a quantum dynamical model.
        A `Dynamics` object combines:

    - A `Propagator`, which specifies how the generator is integrated
      in time to produce finite-time evolution.
    - A `Generator`, which defines the infinitesimal evolution
      (e.g., Schrödinger, von Neumann, or GKSL generator).
    - 'callbacks': a list of callables that accept parameters (state, t)

    This design separates physical law (generator) from numerical
    method (propagator).
    """

    def __init__(
        self,
        propagator: Propagator,
        generator: Generator,
        callbacks: list[Callable] = None,
    ) -> None:
        self._prop = propagator
        self._gen = generator
        if callbacks is not None:
            if not isinstance(callbacks, list) or any(
                [not callable(callback) for callback in callbacks]
            ):
                raise TypeError(
                    f"Callbacks must be a list of type callable, not {type(callbacks)}"
                )
        self._callbacks = callbacks

    def addCallback(self, callback: Callable):
        if callable(callback):
            if self._callbacks is None:
                self._callbacks = [callback]
            else:
                self._callbacks.append(callback)
        else:
            raise TypeError(
                f"Callbacks must be a list of type callable, not {type(callback)}"
            )

    def evolve(
        self, state: QuantumState, ts: Iterable[Real], t0: Real = 0, verbose: int = 0
    ) -> QuantumState:
        """
        Evolve a quantum state from time `t0` to time `t`.

        Parameters
        ----------
        state : QuantumState
            The initial quantum state.
        ts : List[Real]
            List of times to evolve the state to.
        t0 : Real, optional
            Initial time (default is 0).
        verbose: int, optional
            Alters the logging detail provided (default is 0)
        Returns
        -------
        QuantumState
            The final evolved quantum state at t_final = max(ts).
        """
        if min(ts) < t0:
            raise ValueError(f"Evolution time {min(ts)} is less than initial time {t0}")
        if list(ts) != sorted(list(ts)):
            raise ValueError("Ts must be a list of increasing times")

        return self._evolveObject(self._prop.evolve, state, ts, t0, verbose)

    def evolveOperator(
        self, op: Operator, ts: Iterable[Real], t0: Real = 0, verbose: int = 0
    ) -> Operator:
        """
        Evolve an operator in time from `t0` to `ts`. Each ts must be smaller than t0, as Heisenberg evolution works back in time

        Parameters
        ----------
        op : Operator
            Operator to evolve (e.g., observable).
        ts : List[Real]
            List of times to evolve the operator to.
        t0 : Real, optional
            Initial time (default is 0).
        verbose: int, optional
            Alters the logging detail provided (default is 0)
        Returns
        -------
        Operator
            The time-evolved operator.
        """
        if max(ts) > t0:
            raise ValueError(f"Evolution time {max(ts)} is more than initial time {t0}")
        if list(ts) != sorted(list(ts), reverse=True):
            raise ValueError("Ts must be a list of increasing times")

        return self._evolveObject(self._prop.evolveOperator, op, ts, t0, verbose)

    def _evolveObject(
        self, prop_func: Callable, obj, ts: Iterable[Real], t0: Real, verbose: int = 0
    ):
        if not isinstance(ts, Iterable):
            raise TypeError(
                f"Ts must be an iterable of evolution times, not {type(ts)}"
            )

        if verbose == 1:
            ts = tqdm(ts)

        for t in ts:
            obj = prop_func(self._gen, obj, t, t0)
            t0 = t
            self._callback(obj, t)

        return obj

    def _callback(self, op: Any, t: Real) -> None:
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback(op, t)
