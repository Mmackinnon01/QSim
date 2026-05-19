import numba
import numpy as np
import pytest

from qsim.lin_alg import DiscreteTOperator, Operator, TOperator, sigmaX, sigmaZ


@numba.njit
def discrete(t):
    return -1 if t < 2 else 1


def continuous(t):
    return np.sin(t)


@pytest.fixture
def discrete_time_dependent():
    @numba.njit
    def control(t):
        return 1
    return TOperator([(control, sigmaX), (discrete, sigmaZ)])


@pytest.fixture
def discrete_time_dependent_composed():
    return TOperator.from_static(sigmaX) + discrete * TOperator.from_static(sigmaZ)


@pytest.fixture
def all_operations():
    x = TOperator.from_static(sigmaX)
    return x @ sigmaZ + sigmaZ @ x - x @ x + 3 * x


@pytest.fixture
def continuous_time_dependent():
    return TOperator([(lambda t: 1, sigmaX), (continuous, sigmaZ)])


def test_returns_operator(discrete_time_dependent):
    assert isinstance(discrete_time_dependent(0), Operator)


def test_discrete(discrete_time_dependent):
    assert discrete_time_dependent(0) == sigmaX - sigmaZ
    assert discrete_time_dependent(10) == sigmaX + sigmaZ


def test_compile_operator(discrete_time_dependent):
    op = discrete_time_dependent.compile()
    assert pytest.approx(op(0)) == (sigmaX - sigmaZ).matrix

def test_composition(
    discrete_time_dependent, discrete_time_dependent_composed, all_operations
):
    assert discrete_time_dependent(0) == discrete_time_dependent_composed(0)
    assert discrete_time_dependent(10) == discrete_time_dependent_composed(10)
    assert (
        pytest.approx(all_operations(0).matrix)
        == (sigmaX @ sigmaZ + sigmaZ @ sigmaX - sigmaX @ sigmaX + 3 * sigmaX).matrix
    )


def test_continuous(continuous_time_dependent):
    assert continuous_time_dependent(0) == sigmaX
    assert continuous_time_dependent(10) == sigmaX + np.sin(10) * sigmaZ


def test_embed_toperator_in_larger_hilbert_space(discrete_time_dependent):
    assert discrete_time_dependent.changeHilbertSpace(
        new_dims=(2, 2), send_to_sites=(1,)
    )(0) == Operator(np.eye(2)).tensor(sigmaX) - Operator(np.eye(2)).tensor(sigmaZ)


def test_can_be_conjugated(discrete_time_dependent):
    top_conj = discrete_time_dependent.hConj()(0).matrix
    op_conj = (sigmaX.hConj() - sigmaZ.hConj()).matrix
    assert pytest.approx(top_conj) == op_conj


def test_can_be_tensored(discrete_time_dependent):
    top_tensor = discrete_time_dependent.tensor(discrete_time_dependent)(0).matrix
    op_tensor = (sigmaX - sigmaZ).tensor(sigmaX - sigmaZ).matrix
    assert pytest.approx(top_tensor) == op_tensor


def test_xor_tensor(discrete_time_dependent):
    top_tensor = (discrete_time_dependent ^ discrete_time_dependent)(0).matrix
    op_tensor = (sigmaX - sigmaZ).tensor(sigmaX - sigmaZ).matrix
    assert pytest.approx(top_tensor) == op_tensor


def test_can_be_commutated(discrete_time_dependent):
    top_tensor = discrete_time_dependent.commutator(sigmaX)(0).matrix
    op_tensor = (sigmaX - sigmaZ).commutator(sigmaX).matrix
    assert pytest.approx(top_tensor) == op_tensor


def test_change_basis():
    top = sigmaZ + (lambda t: t) * TOperator.from_static(sigmaX)
    top = top.changeBasis(sigmaX.eigenvectors)
    assert (
        pytest.approx(top(1).matrix)
        == (
            (sigmaX.changeBasis(sigmaX.eigenvectors))
            + sigmaZ.changeBasis(sigmaX.eigenvectors)
        ).matrix
    )



def test_discrete_tdop_has_cache():
    op = DiscreteTOperator(
        sigmaZ + (lambda t: 0 if t < 1 else 1) * TOperator.from_static(sigmaX),
        intervals=(1,),
    )
    eval_op = op(0)
    assert pytest.approx(eval_op.matrix) == sigmaZ.matrix
    assert (-1, 1) in op._cache


def test_discrete_matmul():
    op1 = DiscreteTOperator(
        sigmaZ + (lambda t: 0 if t < 1 else 1) * TOperator.from_static(sigmaX),
        intervals=(1,),
    )
    op2 = DiscreteTOperator(
        sigmaZ + (lambda t: 0 if t < 2 else 1) * TOperator.from_static(sigmaX),
        intervals=(2,),
    )
    op_matmul = op1 @ op2

    assert op_matmul._intervals == (1, 2)
    assert pytest.approx(op_matmul(0).matrix) == (op1(0) @ op2(0)).matrix
    assert pytest.approx(op_matmul(1).matrix) == (op1(1) @ op2(1)).matrix
    assert pytest.approx(op_matmul(2).matrix) == (op1(2) @ op2(2)).matrix
    assert isinstance(op_matmul, DiscreteTOperator)
    assert isinstance(op1 @ TOperator.from_static(sigmaX), TOperator)


def test_discrete_tensor():
    op1 = DiscreteTOperator(
        sigmaZ + (lambda t: 0 if t < 1 else 1) * TOperator.from_static(sigmaX),
        intervals=(1,),
    )
    op2 = DiscreteTOperator(
        sigmaZ + (lambda t: 0 if t < 2 else 1) * TOperator.from_static(sigmaX),
        intervals=(2,),
    )
    op_tensor = op1 ^ op2

    assert op_tensor._intervals == (1, 2)
    assert pytest.approx(op_tensor(0).matrix) == (op1(0) ^ op2(0)).matrix
    assert pytest.approx(op_tensor(1).matrix) == (op1(1) ^ op2(1)).matrix
    assert pytest.approx(op_tensor(2).matrix) == (op1(2) ^ op2(2)).matrix
    assert isinstance(op_tensor, DiscreteTOperator)
    assert isinstance(op1 ^ TOperator.from_static(sigmaX), TOperator)

def test_compile_discrete_operator():
    @numba.njit
    def control(t):
        return 0 if t < 1 else 1
    op1 = DiscreteTOperator(
        sigmaZ + control * TOperator.from_static(sigmaX),
        intervals=(1,),
    )
    op = op1.compile()
    assert pytest.approx(op(0)) == sigmaZ.matrix