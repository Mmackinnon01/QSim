import numpy as np
import pytest

from qsim.lin_alg import Operator, sigmaX
from qsim.state import DensityMatrix, Ket


@pytest.fixture
def bell_state():
    return DensityMatrix(
        state=np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) * 0.5
    )


@pytest.fixture
def qubit_state():
    return DensityMatrix(state=np.array([[0.5, 0.5j], [-0.5j, 0.5]]))


@pytest.fixture
def maximally_mixed_state():
    return DensityMatrix(np.array([[0.5, 0], [0, 0.5]]))


def test_trace(bell_state):
    assert bell_state.trace() == 1


def test_matmul(qubit_state):
    assert qubit_state @ qubit_state == qubit_state


def test_matmul_by_non_matrix(qubit_state):
    with pytest.raises(TypeError):
        qubit_state @ 2


def test_purity(bell_state, maximally_mixed_state):
    assert np.isclose(bell_state.purity(), 1)
    assert np.isclose(maximally_mixed_state.purity(), 0.5)


def test_dm_plus_op_gives_dm(qubit_state):
    assert isinstance(qubit_state + sigmaX, DensityMatrix)


def test_can_cast_operator_to_density_if_legit():
    assert DensityMatrix(Operator(np.array([[1, 0], [0, 0]]))) == DensityMatrix(
        np.array([[1, 0], [0, 0]])
    )


def test_tensor():
    s = DensityMatrix(np.array([[1, 0], [0, 0]]))
    assert (s ^ s).state == pytest.approx(
        np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )
