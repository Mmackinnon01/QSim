import numpy as np
import pytest

from qsim.state import Bra, DensityMatrix, Ket
from qsim.state.wave_vector import WaveVector


@pytest.fixture
def bell_wave_vector():
    return Ket(np.array([1, 0, 0, 1]) / 2**0.5)


@pytest.fixture
def complex_wave_vector():
    return Ket(np.array([1 / 2**0.5, (1 / 2**0.5) * 1j]))


@pytest.fixture
def bell_density_matrix():
    return DensityMatrix(
        np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    )


def test_hermConj(bell_wave_vector, complex_wave_vector):
    assert isinstance(bell_wave_vector, Ket)
    assert isinstance(bell_wave_vector.hConj(), Bra)
    assert np.allclose(
        complex_wave_vector.hConj().matrix,
        np.array([1 / 2**0.5, -(1 / 2**0.5) * 1j]).reshape(1, -11),
    )


def test_inner_product(complex_wave_vector):
    assert np.isclose(complex_wave_vector.hConj() @ complex_wave_vector, 1)


def test_outer_product(complex_wave_vector):
    dm = complex_wave_vector @ complex_wave_vector.hConj()
    assert pytest.approx(dm.matrix) == np.array([[0.5, -0.5j], [0.5j, 0.5]])


def test_project_onto_density(bell_wave_vector, bell_density_matrix):
    print(bell_wave_vector.hConj().matrix.shape)
    print(bell_density_matrix.matrix.shape)
    left_project = bell_wave_vector.hConj() @ bell_density_matrix
    right_project = bell_density_matrix @ bell_wave_vector
    assert pytest.approx(left_project.matrix) == bell_wave_vector.matrix.reshape(1, -1)
    assert pytest.approx(right_project.matrix) == bell_wave_vector.matrix


def test_mul(bell_wave_vector):
    assert (
        pytest.approx((((2 - 1j) * bell_wave_vector).matrix))
        == np.array([2 - 1j, 0, 0, 2 - 1j]).reshape(-1, 1) / 2**0.5
    )
    assert isinstance(bell_wave_vector * 2, Ket)
    assert isinstance(2 * bell_wave_vector, Ket)
    assert isinstance(bell_wave_vector.hConj() * 2, Bra)
    assert isinstance(2 * bell_wave_vector.hConj(), Bra)


def test_normalisation(bell_wave_vector):
    bell_wave_vector_normalised = bell_wave_vector.normalise()
    assert not (2 * bell_wave_vector).isNormalised()
    assert bell_wave_vector_normalised.isNormalised()
    assert pytest.approx(bell_wave_vector_normalised.norm()) == 1


def test_tensor():
    b = Ket(np.array([0, 1]))
    c = Ket(np.array([1, 1]) / 2**0.5)

    assert (
        pytest.approx(b.tensor(c).matrix)
        == np.array([0, 0, 1, 1]).reshape(-1, 1) / 2**0.5
    )
    assert (
        pytest.approx((b ^ c).matrix) == np.array([0, 0, 1, 1]).reshape(-1, 1) / 2**0.5
    )


def test_tensor_ket_bra_incompatible():
    a = Bra(np.array([1, 0]))
    b = Ket(np.array([1, 0]))
    a.tensor(b) == NotImplemented


def test_partial_trace(bell_wave_vector):
    a = Ket(np.array([1, 0]))
    b = Ket(np.array([0, 1]))
    c = Ket(np.array([1, 1]) / 2**0.5)
    assert (
        pytest.approx(a.tensor(b).tensor(c).partialTrace((2, 2, 2), (1, 0)).matrix)
        == (b @ b.hConj()).tensor(a @ a.hConj()).matrix
    )
