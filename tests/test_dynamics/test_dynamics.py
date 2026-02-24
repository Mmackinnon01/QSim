import math

import numpy as np
import pytest

from qsim.dynamics import (
    ExponentialPropagator,
    GKSLGenerator,
    HamiltonianGenerator,
    RK4Propagator,
)
from qsim.dynamics.base import Dynamics
from qsim.operator import Observable, Operator, sigmaPlus, sigmaX, sigmaZ
from qsim.state import Bra, DensityMatrix, Ket

PI = math.pi

spin_down = DensityMatrix(np.array([[1, 0], [0, 0]]))
zObservable = Observable(sigmaZ.matrix)

generator = GKSLGenerator(H=sigmaX, jumps=[sigmaPlus])
unitary_dynamic = GKSLGenerator(H=sigmaX, jumps=[])

propagator = RK4Propagator()
dynamics = Dynamics(propagator, unitary_dynamic)


def test_evolve_dm_two_step():
    assert pytest.approx(
        dynamics.evolve(spin_down, ts=[PI / 4, PI / 2], t0=0).matrix
    ) == np.array([[0, 0], [0, 1]])


def test_evolve_dm_callback():
    state = 0
    t = 0

    def callback(s, x):
        nonlocal state
        state = s
        nonlocal t
        t = x

    dynamics = Dynamics(propagator, unitary_dynamic, callbacks=[callback])
    dynamics.evolve(spin_down, ts=[PI / 2])
    assert pytest.approx(state.matrix) == np.array([[0, 0], [0, 1]])
    assert t == PI / 2


def test_evolve_operator_two_step():
    assert pytest.approx(
        dynamics.evolve(spin_down, ts=[PI / 4, PI / 2], t0=0).matrix
    ) == np.array([[0, 0], [0, 1]])


def test_evolve_operator_callback():
    op = 0
    t = 0

    def callback(o, x):
        nonlocal op
        op = o
        nonlocal t
        t = x

    dynamics = Dynamics(propagator, unitary_dynamic, callbacks=[callback])
    dynamics.evolveOperator(sigmaX, ts=[PI / 2], t0=PI)
    assert pytest.approx(op.matrix) == sigmaX.matrix
    assert t == PI / 2


def test_invalid_times():
    with pytest.raises(ValueError):
        dynamics.evolveOperator(sigmaX, ts=[PI / 2], t0=0)
    with pytest.raises(ValueError):
        dynamics.evolve(spin_down, ts=[PI / 2], t0=PI)


def test_unsorted_times_invalid():
    with pytest.raises(ValueError):
        dynamics.evolve(spin_down, ts=[PI / 2, PI / 4], t0=0)
    with pytest.raises(ValueError):
        dynamics.evolveOperator(sigmaX, ts=[PI / 2, PI], t0=2 * PI)


def test_non_iterable_ts():
    with pytest.raises(TypeError):
        dynamics.evolve(spin_down, ts=PI)
