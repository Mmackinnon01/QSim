from email.mime import base
from tkinter import TOP
from token import OP

import numpy as np
import pytest

from qsim.operator import Operator, TOperator, sigmaX, sigmaZ


def discrete(t):
    return -1 if t < 2 else 1


def continuous(t):
    return np.sin(t)


@pytest.fixture
def discrete_time_dependent():
    return TOperator([(lambda t: 1, sigmaX), (discrete, sigmaZ)])


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


def test_compoosition(
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


def test_can_be_commutated(discrete_time_dependent):
    top_tensor = discrete_time_dependent.commutator(sigmaX)(0).matrix
    op_tensor = (sigmaX - sigmaZ).commutator(sigmaX).matrix
    assert pytest.approx(top_tensor) == op_tensor
