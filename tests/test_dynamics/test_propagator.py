import math

import numpy as np
import pytest

from qsim.dynamics import (
    ExponentialPropagator,
    GKSLGenerator,
    HamiltonianGenerator,
    RK4Propagator,
)
from qsim.operator import Observable, Operator, sigmaPlus, sigmaX, sigmaZ
from qsim.state import Bra, DensityMatrix, Ket

PI = math.pi

spin_down = DensityMatrix(np.array([[1, 0], [0, 0]]))
zObservable = Observable(sigmaZ.matrix)

generator = GKSLGenerator(H=sigmaX, jumps=[sigmaPlus])
unitary_dynamic = GKSLGenerator(H=sigmaX, jumps=[])

propagator = RK4Propagator()


def test_evolve_dm():
    assert pytest.approx(
        propagator.evolve(unitary_dynamic, spin_down, t=PI / 2).matrix
    ) == np.array([[0, 0], [0, 1]])


def test_evolve_observable():
    assert (
        pytest.approx(
            propagator.evolveOperator(unitary_dynamic, zObservable, t=PI / 2).matrix
        )
        == -zObservable.matrix
    )


def test_ts_changes():
    assert RK4Propagator(ts=0.1).ts == 0.1


sigmaX = Operator(np.array([[0, 1], [1, 0]]))
zObservable = Observable(np.array([[1, 0], [0, -1]]))

spinUpDensityMatrix = DensityMatrix(np.array([[1, 0], [0, 0]]))
spinDownDensityMatrix = DensityMatrix(np.array([[0, 0], [0, 1]]))

spinUpKet = Ket(np.array([1, 0]))
spinDownKet = Ket(np.array([0, 1]))

spinUpBra = Bra(np.array([1, 0]))
spinDownBra = Bra(np.array([0, 1]))

xSpinGenerator = HamiltonianGenerator(H=sigmaX)
exponential_propagator = ExponentialPropagator()

PI = math.pi


def test_evolve_density_matrix():
    evolved_state = exponential_propagator.evolve(
        xSpinGenerator, spinUpDensityMatrix, t=PI / 2
    )
    assert pytest.approx(evolved_state.state) == spinDownDensityMatrix.state
    assert isinstance(evolved_state, DensityMatrix)


def test_evolve_ket():
    evolved_state = exponential_propagator.evolve(xSpinGenerator, spinUpKet, t=PI / 2)
    assert pytest.approx(evolved_state.state) == (-1j * spinDownKet).state
    assert isinstance(evolved_state, Ket)


def test_evolve_bra():
    evolved_state = exponential_propagator.evolve(xSpinGenerator, spinUpBra, t=PI / 2)
    assert pytest.approx(evolved_state.state) == (1j * spinDownBra).state
    assert isinstance(evolved_state, Bra)


def test_evolve_operator():
    evolved_observable = exponential_propagator.evolveOperator(
        xSpinGenerator, zObservable, t=PI / 2
    )
    assert pytest.approx(evolved_observable.matrix) == (-1 * zObservable).matrix
    assert isinstance(evolved_observable, Operator)
