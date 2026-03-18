import math

import numpy as np
import pytest

from qsim.dynamics import ExponentialPropagator, HamiltonianGenerator
from qsim.lin_alg import Observable, Operator, sigmaX
from qsim.state import Bra, DensityMatrix, Ket

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
        xSpinGenerator, spinUpDensityMatrix, t_final=PI / 2
    )
    assert pytest.approx(evolved_state.state) == spinDownDensityMatrix.state
    assert isinstance(evolved_state, DensityMatrix)


def test_evolve_ket():
    evolved_state = exponential_propagator.evolve(
        xSpinGenerator, spinUpKet, t_final=PI / 2
    )
    assert pytest.approx(evolved_state.matrix) == (-1j * spinDownKet).matrix
    assert isinstance(evolved_state, Ket)


def test_evolve_bra():
    evolved_state = exponential_propagator.evolve(
        xSpinGenerator, spinUpBra, t_final=PI / 2
    )
    assert pytest.approx(evolved_state.matrix) == (1j * spinDownBra).matrix
    assert isinstance(evolved_state, Bra)


def test_evolve_operator():
    evolved_observable = exponential_propagator.evolveOperator(
        xSpinGenerator, zObservable, t_final=PI / 2
    )
    assert pytest.approx(evolved_observable.matrix) == (-1 * zObservable).matrix
    assert isinstance(evolved_observable, Operator)
