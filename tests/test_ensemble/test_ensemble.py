import numpy as np
import pytest

from qsim.ensemble import Haar, HilbertSchmidt
from qsim.operator.base import Operator
from qsim.state.density_matrix import DensityMatrix
from qsim.state.wave_vector import Ket


def test_generate_haar_wave_vector():
    state = Haar.generateKet(4)
    assert isinstance(state, Ket)
    assert state.dim == 4


def test_generate_haar_batch_ket():
    states = Haar.generateKetBatch(4, n=10)
    assert isinstance(states[0], Ket)
    assert states[0].dim == 4
    assert len(states) == 10


def test_generate_haar_density():
    state = Haar.generateDM(4)
    assert isinstance(state, DensityMatrix)
    assert state.dim == 4
    assert pytest.approx(state.purity()) == 1


def test_generate_haar_batch_dm():
    states = Haar.generateDMBatch(4, n=10)
    assert isinstance(states[0], DensityMatrix)
    assert states[0].dim == 4
    assert pytest.approx(states[0].purity()) == 1
    assert len(states) == 10


def test_generate_haar_unitary():
    state = Haar.generateUnitary(4)
    assert isinstance(state, Operator)
    assert state.dim == 4
    assert state.hConj() @ state == Operator(np.eye(4))


def test_generate_haar_batch_unitary():
    states = Haar.generateUnitaryBatch(4, n=5)
    assert isinstance(states[0], Operator)
    assert states[0].dim == 4
    assert states[0].hConj() @ states[0] == Operator(np.eye(4))
    assert len(states) == 5


def test_generate_hs_density():
    state = HilbertSchmidt.generateDM(4)
    assert isinstance(state, DensityMatrix)
    assert state.dim == 4


def test_generate_hs_density_rank():
    state = HilbertSchmidt.generateDM(4, 1)
    assert isinstance(state, DensityMatrix)
    assert state.dim == 4
    assert pytest.approx(state.purity()) == 1


def test_generate_hs_batch_dm():
    states = HilbertSchmidt.generateDMBatch(4, 1, n=10)
    assert isinstance(states[0], DensityMatrix)
    assert states[0].dim == 4
    assert pytest.approx(states[0].purity()) == 1
    assert len(states) == 10
