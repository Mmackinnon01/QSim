from multiprocessing import Value
from token import OP
from typing import Type

import numpy as np
import pytest

from qsim.lin_alg import Observable, Operator

pauliZ = Operator(np.array([[1, 0], [0, -1]]))
pauliX = Operator(np.array([[0, 1], [1, 0]]))


pauliZX = Operator(np.kron(np.array([[1, 0], [0, -1]]), np.array([[0, 1], [1, 0]])))

import numpy as np
import pytest


@pytest.fixture
def op():
    return Operator(np.array([[0.5, 0.5j], [-0.5j, 0.5]]))


def test_hermitian_conjugate(op):
    assert op.hConj() == op


def test_complex_mul(op):
    assert op * (-1 + 1j) == Operator(
        np.array([[-0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, -0.5 + 0.5j]])
    )


def test_mul_by_density_matrix_invalid(op):
    with pytest.raises(TypeError):
        op * op


def test_div_by_density_matrix_invalid(op):
    with pytest.raises(TypeError):
        op / op


def test_add_matrices(op):
    assert pytest.approx((op + op).matrix) == np.array([[1, 1j], [-1j, 1]])


def test_subtract_matrices(op):
    assert pytest.approx((op - op).matrix) == np.array([[0, 0], [0, 0]])


def test_tensor(op):
    assert pytest.approx(op.tensor(op).matrix) == np.kron(op.matrix, op.matrix)
    assert pytest.approx((op ^ op).matrix) == np.kron(op.matrix, op.matrix)


def test_tensor_invalid_type(op):
    op.tensor(1) == NotImplemented


def test_operator_commutator():
    assert pytest.approx(pauliZ.commutator(pauliX).matrix) == np.array(
        [[0, 2], [-2, 0]]
    )


def test_pow():
    assert pytest.approx((pauliZ**2).matrix) == np.array([[1, 0], [0, 1]])


def test_change_hilbert_space_single_op():
    assert pytest.approx(
        pauliZ.changeHilbertSpace(
            new_dims=(2, 2), base_dims=(2,), send_to_sites=(1,)
        ).matrix
    ) == np.kron(np.eye(2), pauliZ.matrix)
    assert pytest.approx(
        pauliX.changeHilbertSpace(new_dims=(2, 4), send_to_sites=(0,)).matrix
    ) == np.kron(pauliX.matrix, np.eye(4))


def test_change_hilbert_space_dual_op():
    assert pytest.approx(
        pauliZX.changeHilbertSpace(
            new_dims=(2, 4, 2), base_dims=(2, 2), send_to_sites=(2, 0)
        ).matrix
    ) == np.kron(pauliX.matrix, np.kron(np.eye(4), pauliZ.matrix))


def test_change_hilbert_space_invalid_args():
    with pytest.raises(ValueError):
        pauliZX.changeHilbertSpace(
            new_dims=(4, 2), base_dims=(2, 2), send_to_sites=(2, 0)
        )
    with pytest.raises(ValueError):
        pauliZX.changeHilbertSpace(
            new_dims=(2, 4, 2), base_dims=(2, 2), send_to_sites=(-1, 0)
        )
    with pytest.raises(ValueError):
        pauliZX.changeHilbertSpace(
            new_dims=(2, 4, 2), base_dims=(2, 2), send_to_sites=(2, 2)
        )
    with pytest.raises(ValueError):
        pauliZX.changeHilbertSpace(
            new_dims=(2, 4, 2), base_dims=(4, 2), send_to_sites=(2, 0)
        )


def test_partial_trace():
    assert pytest.approx(pauliZX.partialTrace((2, 2), (1,)).matrix) == np.array(
        [[0, 0], [0, 0]]
    )


def test_callable_returns_self():
    assert pauliZ(3) == pauliZ


def test_callable_returns_value_error_if_not_float():
    with pytest.raises(TypeError):
        pauliZ("3")


def test_partial_trace_reorder():
    a = Operator(np.array([[1, 0], [0, 0]]))
    b = Operator(np.array([[0, 0], [0, 1]]))
    c = Operator(np.array([[0.5, 0], [0, 0.5]]))
    assert (
        pytest.approx(a.tensor(b).tensor(c).partialTrace((2, 2, 2), (1, 0)).matrix)
        == b.tensor(a).matrix
    )


def test_transpose():
    assert pytest.approx(Operator(np.array([[1, 2], [3, 4]])).T.matrix) == np.array(
        [[1, 3], [2, 4]]
    )


def test_conjugate():
    assert pytest.approx(
        Operator(np.array([[1 + 1j, 2 - 2j], [3, 4]])).conj().matrix
    ) == np.array([[1 - 1j, 2 + 2j], [3, 4]])


def test_eigenbasis():
    assert pytest.approx(pauliX.eigenvectors) == np.array(
        [[-(0.5**0.5), (0.5**0.5)], [(0.5**0.5), (0.5**0.5)]]
    )
    assert pytest.approx(pauliX.eigenvalues) == np.array([1, -1]) or pytest.approx(
        pauliX.eigenvalues
    ) == np.array([-1, 1])


def test_change_basis():
    assert pytest.approx(pauliX.changeBasis(pauliX.eigenvectors).matrix) == np.array(
        [[-1, 0], [0, 1]]
    )
