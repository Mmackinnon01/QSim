import math

import numpy as np
import pytest

from qsim.dynamics import GKSLGenerator, RK4Propagator
from qsim.lin_alg import Observable, sigmaPlus, sigmaX, sigmaZ
from qsim.state import DensityMatrix, Ket

PI = math.pi

spin_down = DensityMatrix(np.array([[1, 0], [0, 0]]))
zObservable = Observable(sigmaZ.matrix)

generator = GKSLGenerator(H=sigmaX, jumps=[sigmaPlus])
unitary_dynamic = GKSLGenerator(H=sigmaX, jumps=[])

propagator = RK4Propagator()


def test_evolve_dm():
    assert pytest.approx(
        propagator.evolve(unitary_dynamic, spin_down, t_final=PI / 2).matrix
    ) == np.array([[0, 0], [0, 1]])


def test_evolve_observable():
    assert (
        pytest.approx(
            propagator.evolveOperator(
                unitary_dynamic, zObservable, t_final=0, t0=PI / 2
            ).matrix
        )
        == -zObservable.matrix
    )


def test_ts_changes():
    assert RK4Propagator(ts=0.1).ts == 0.1
