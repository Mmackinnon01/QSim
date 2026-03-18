import math
from re import L

import numpy as np
import pytest

from qsim.dynamics import DiagonalPropagator, HamiltonianGenerator
from qsim.lin_alg import Observable, Operator, sigmaX
from qsim.state import Bra, DensityMatrix, Ket

PI = math.pi

sigmaX = Operator(np.array([[0, 1], [1, 0]]))
zObservable = Observable(np.array([[1, 0], [0, -1]])).changeBasis(sigmaX.eigenvectors)

spinUpDensityMatrix = DensityMatrix(np.array([[1, 0], [0, 0]])).changeBasis(
    sigmaX.eigenvectors
)
spinDownDensityMatrix = DensityMatrix(np.array([[0, 0], [0, 1]])).changeBasis(
    sigmaX.eigenvectors
)

spinUpKet = Ket(np.array([1, 0])).changeBasis(sigmaX.eigenvectors)
spinDownKet = Ket(np.array([0, 1])).changeBasis(sigmaX.eigenvectors)

spinUpBra = Bra(np.array([1, 0])).changeBasis(sigmaX.eigenvectors)
spinDownBra = Bra(np.array([0, 1])).changeBasis(sigmaX.eigenvectors)

xSpinGenerator = HamiltonianGenerator(H=sigmaX.changeBasis(sigmaX.eigenvectors))
xSpinGenerator_non_diagonal = HamiltonianGenerator(H=sigmaX)

PI = math.pi

diagonal_propagator = DiagonalPropagator()


def test_non_diagonal_H_invalid():
    with pytest.raises(ValueError):
        diagonal_propagator.evolve(
            xSpinGenerator_non_diagonal, spinUpDensityMatrix, t_final=PI / 2
        )


def test_evolve_density_matrix():
    evolved_state = diagonal_propagator.evolve(
        xSpinGenerator, spinUpDensityMatrix, t_final=PI / 2
    )
    assert pytest.approx(evolved_state.state) == spinDownDensityMatrix.state
    assert isinstance(evolved_state, DensityMatrix)


def test_evolve_ket():
    evolved_state = diagonal_propagator.evolve(
        xSpinGenerator, spinUpKet, t_final=PI / 2
    )
    assert pytest.approx(evolved_state.matrix) == (-1j * spinDownKet).matrix
    assert isinstance(evolved_state, Ket)


def test_evolve_bra():
    evolved_state = diagonal_propagator.evolve(
        xSpinGenerator, spinUpBra, t_final=PI / 2
    )
    assert pytest.approx(evolved_state.matrix) == (1j * spinDownBra).matrix
    assert isinstance(evolved_state, Bra)


def test_evolve_operator():
    evolved_observable = diagonal_propagator.evolveOperator(
        xSpinGenerator, zObservable, t_final=PI / 2
    )
    assert pytest.approx(evolved_observable.matrix) == (-1 * zObservable).matrix
    assert isinstance(evolved_observable, Operator)
