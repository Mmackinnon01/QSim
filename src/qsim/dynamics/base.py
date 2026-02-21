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

from ast import Call
from numbers import Real
from typing import Any, Callable

from qsim.dynamics.generator import Generator
from qsim.dynamics.propagator import Propagator
from qsim.operator import Operator
from qsim.operator.base import Operator
from qsim.state import QuantumState
from qsim.state.base import QuantumState


class Dynamics:
    """
    High-level wrapper representing a quantum dynamical model.
        A `Dynamics` object combines:

    - A `Generator`, which defines the infinitesimal evolution
      (e.g., Schrödinger, von Neumann, or GKSL generator).
    - A `Propagator`, which specifies how the generator is integrated
      in time to produce finite-time evolution.

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
            if not isinstance(callbacks, list) or not callable(callbacks[0]):
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
                f"Callbacks must be a list of type callable, not {type(callbacks)}"
            )

    def evolve(self, state: QuantumState, ts: list[Real], t0: Real = 0) -> QuantumState:
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
        Returns
        -------
        QuantumState
            The final evolved quantum state at t_final = max(ts).
        """
        if not isinstance(ts, list):
            raise TypeError(f"Ts must be a list of evolution times, not {type(ts)}")
        if min(ts) < t0:
            raise ValueError(f"Evolution time {min(ts)} is less than initial time {t0}")

        for t in sorted(ts):
            state = self._prop.evolve(self._gen, state, t - t0, t0)
            t0 = t
            self._callback(state, t)

        return state

    def evolveOperator(self, op: Operator, ts: list[Real], t0: Real = 0) -> Operator:
        """
        Evolve an operator in time from `t0` to `t0-t`.

        Parameters
        ----------
        op : Operator
            Operator to evolve (e.g., observable).
        ts : List[Real]
            List of times to evolve the operator to.
        t0 : Real, optional
            Initial time (default is 0).
        Returns
        -------
        Operator
            The time-evolved operator.
        """

        if not isinstance(ts, list):
            raise TypeError(f"Ts must be a list of evolution times, not {type(ts)}")
        if max(ts) > t0:
            raise ValueError(
                f"Evolution time {max(ts)} is greater than initial time {t0}"
            )

        for t in sorted(ts, reverse=True):
            op = self._prop.evolveOperator(self._gen, op, t0 - t, t0)
            t0 = t
            self._callback(op, t)

        return op

    def _callback(self, op: Any, t: Real) -> None:
        if self._callbacks is not None:
            for callback in self._callbacks:
                callback(op, t)
