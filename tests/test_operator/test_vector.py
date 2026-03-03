from numbers import Number

import numpy as np
import pytest

from qsim.lin_alg import Operator, Vector


@pytest.fixture
def bell_wave_vector():
    return Vector(np.array([1, 0, 0, 1]) / 2**0.5)


@pytest.fixture
def complex_wave_vector():
    return Vector(np.array([1 / 2**0.5, (1 / 2**0.5) * 1j]))


def test_conj(bell_wave_vector, complex_wave_vector):
    assert isinstance(bell_wave_vector, Vector)
    assert isinstance(bell_wave_vector.conj(), Vector)
    assert np.allclose(
        complex_wave_vector.conj().matrix,
        np.array([[1 / 2**0.5, -(1 / 2**0.5) * 1j]]).reshape(-1, 1),
    )


def test_norm(complex_wave_vector):
    assert np.isclose(complex_wave_vector.norm(), 1)


def test_outer_product(complex_wave_vector):
    dm = complex_wave_vector @ complex_wave_vector.hConj()
    assert isinstance(dm, Operator)
    assert pytest.approx(dm.matrix) == np.array([[0.5, -0.5j], [0.5j, 0.5]])


def test_inner_product(complex_wave_vector):
    dm = complex_wave_vector.hConj() @ complex_wave_vector
    assert isinstance(dm, Number)
    assert pytest.approx(dm) == 1


def test_tensor():
    v = Vector(np.array([1, 0]))
    assert pytest.approx((v ^ v).matrix) == np.array([[1, 0, 0, 0]]).reshape(-1, 1)


def test_mul(bell_wave_vector):
    assert (
        pytest.approx((((2 - 1j) * bell_wave_vector).matrix))
        == np.array([[2 - 1j, 0, 0, 2 - 1j]]).reshape(-1, 1) / 2**0.5
    )
    assert isinstance(bell_wave_vector * 2, Vector)
    assert isinstance(2 * bell_wave_vector, Vector)
