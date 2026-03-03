import math

import numpy as np
import pytest

from qsim.dynamics import GKSLGenerator, HamiltonianGenerator
from qsim.lin_alg import (
    Observable,
    Operator,
    sigmaMinus,
    sigmaPlus,
    sigmaX,
    sigmaY,
    sigmaZ,
)
from qsim.lin_alg.time_dependent_operator import TOperator
from qsim.state import Bra, DensityMatrix, Ket

PI = math.pi

spin_down = DensityMatrix(np.array([[1, 0], [0, 0]]))
zObservable = Observable(sigmaZ.matrix)

dynamic = GKSLGenerator(H=sigmaX, jumps=[sigmaPlus])
unitary_dynamic = GKSLGenerator(H=sigmaX, jumps=[])


def test_derivative_dm():
    assert pytest.approx(dynamic.onState(spin_down).matrix) == np.array(
        [[-1, 1j], [-1j, 1]]
    )


def test_derivative_observable():
    assert pytest.approx(dynamic.onOperator(zObservable).matrix) == np.array(
        [[-2, -2j], [2j, 0]]
    )


def test_wavevector_invalid():
    with pytest.raises(TypeError):
        dynamic.onState(Ket(np.array([1, 0])))
        dynamic.onState(Bra(np.array([1, 0])))


single_qubit_x = GKSLGenerator(H=sigmaX, jumps=[sigmaPlus])
single_qubit_y = GKSLGenerator(H=sigmaY, jumps=[sigmaMinus])

two_qubit_xx = GKSLGenerator(H=sigmaX.tensor(sigmaX))

dual_spin_down = DensityMatrix(
    np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
)


@pytest.fixture
def composite():
    x_embedded = single_qubit_x.changeHilbertSpace((2, 2), send_to_sites=(0,))
    y_embedded = single_qubit_y.changeHilbertSpace((2, 2), send_to_sites=(1,))
    return two_qubit_xx + x_embedded + y_embedded


@pytest.fixture
def manual_composite_H():
    return (
        np.kron(sigmaX.matrix, np.eye(2))
        + np.kron(np.eye(2), sigmaY.matrix)
        + np.kron(sigmaX.matrix, sigmaX.matrix)
    )


@pytest.fixture
def simple_composite():
    x_no_damping = GKSLGenerator(H=sigmaX, jumps=[])
    x_embedded1 = x_no_damping.changeHilbertSpace((2, 2), send_to_sites=(0,))
    x_embedded2 = x_no_damping.changeHilbertSpace((2, 2), send_to_sites=(1,))
    return x_embedded1 + x_embedded2


def test_add_dynamics(composite):
    assert pytest.approx(composite.H.matrix) == np.kron(
        np.eye(2), sigmaY.matrix
    ) + np.kron(sigmaX.matrix, np.eye(2)) + np.kron(sigmaX.matrix, sigmaX.matrix)


def test_derivative(composite, manual_composite_H):
    state = dual_spin_down.state
    expected_unitary = -1j * (manual_composite_H @ state - state @ manual_composite_H)
    jump1 = sigmaPlus.tensor(Operator(np.eye(2))).matrix
    jump2 = Operator(np.eye(2)).tensor(sigmaMinus).matrix
    expected_dissipative = sum(
        [
            jump @ state @ jump.conj().T
            - 0.5 * (jump.conj().T @ jump @ state + state @ jump.conj().T @ jump)
            for jump in [jump1, jump2]
        ]
    )
    assert (
        pytest.approx(composite.onState(dual_spin_down).matrix, abs=10**-10)
        == expected_unitary + expected_dissipative
    )


def test_observable_derivative(composite, manual_composite_H):
    ob = sigmaZ.tensor(sigmaZ).matrix
    expected_unitary = 1j * (manual_composite_H @ ob - ob @ manual_composite_H)
    jump1 = sigmaPlus.tensor(Operator(np.eye(2))).matrix
    jump2 = Operator(np.eye(2)).tensor(sigmaMinus).matrix
    expected_dissipative = sum(
        [
            jump.conj().T @ ob @ jump
            - 0.5 * (jump.conj().T @ jump @ ob + ob @ jump.conj().T @ jump)
            for jump in [jump1, jump2]
        ]
    )
    assert (
        pytest.approx(composite.onOperator(sigmaZ.tensor(sigmaZ)).matrix)
        == expected_unitary + expected_dissipative
    )


sigmaX = Operator(np.array([[0, 1], [1, 0]]))
zObservable = Observable(np.array([[1, 0], [0, -1]]))

spinUpDensityMatrix = DensityMatrix(np.array([[1, 0], [0, 0]]))
spinDownDensityMatrix = DensityMatrix(np.array([[0, 0], [0, 1]]))

spinUpKet = Ket(np.array([1, 0]))
spinDownKet = Ket(np.array([0, 1]))

spinUpBra = Bra(np.array([1, 0]))
spinDownBra = Bra(np.array([0, 1]))

xSpinDynamics = HamiltonianGenerator(H=sigmaX)

PI = math.pi


def test_generator_density_matrix():
    H = xSpinDynamics.onState(spinUpDensityMatrix)
    assert (
        pytest.approx(H.matrix) == -1j * sigmaX.commutator(spinUpDensityMatrix).matrix
    )


def test_generator_ket():
    H = xSpinDynamics.onState(spinUpKet).matrix
    assert pytest.approx(H) == -1j * (sigmaX @ spinUpKet).matrix


def test_generator_bra():
    H = xSpinDynamics.onState(spinUpBra).matrix
    assert pytest.approx(H) == 1j * (spinUpBra @ sigmaX.hConj()).matrix


def test_evolve_operator():
    H = xSpinDynamics.onOperator(zObservable).matrix
    assert pytest.approx(H) == 1j * sigmaX.commutator(zObservable).matrix


single_qubit_x_hamiltonian = HamiltonianGenerator(H=sigmaX)
single_qubit_y_hamiltonian = HamiltonianGenerator(H=sigmaY)

two_qubit_xx_hamiltonian = HamiltonianGenerator(H=sigmaX.tensor(sigmaX))

bell_state_dm = DensityMatrix(
    np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) * 0.5
)
bell_state_ket = Ket(np.array([1, 0, 0, 1]) / 2**0.5)

PI = math.pi


@pytest.fixture
def composite_hamiltonian():
    x_embedded_hamiltonian = single_qubit_x_hamiltonian.changeHilbertSpace(
        new_dims=(2, 2), send_to_sites=(0,)
    )
    y_embedded_hamiltonian = single_qubit_y_hamiltonian.changeHilbertSpace(
        new_dims=(2, 2), send_to_sites=(1,)
    )
    return two_qubit_xx_hamiltonian + x_embedded_hamiltonian + y_embedded_hamiltonian


@pytest.fixture
def manual_composite_H():
    return (
        np.kron(sigmaX.matrix, np.eye(2))
        + np.kron(np.eye(2), sigmaY.matrix)
        + np.kron(sigmaX.matrix, sigmaX.matrix)
    )


@pytest.fixture
def simple_composite_hamiltonian():
    x_embedded = single_qubit_x_hamiltonian.changeHilbertSpace(
        new_dims=(2, 2), send_to_sites=(0,)
    )
    y_embedded = single_qubit_y_hamiltonian.changeHilbertSpace(
        new_dims=(2, 2), send_to_sites=(1,)
    )
    return x_embedded + y_embedded


def test_add_dynamics(composite_hamiltonian):
    assert pytest.approx(composite_hamiltonian.H.matrix) == np.kron(
        np.eye(2), sigmaY.matrix
    ) + np.kron(sigmaX.matrix, np.eye(2)) + np.kron(sigmaX.matrix, sigmaX.matrix)


def test_generator_separable_dynamics_ket(simple_composite_hamiltonian):
    g = simple_composite_hamiltonian.onState(bell_state_ket)
    assert (
        pytest.approx(g.matrix)
        == -1j
        * (np.kron(np.eye(2), sigmaY.matrix) + np.kron(sigmaX.matrix, np.eye(2)))
        @ bell_state_ket.matrix
    )


def test_evolve_density_matrix(composite_hamiltonian, manual_composite_H):
    evolved_state = composite_hamiltonian.onState(bell_state_dm)
    assert pytest.approx(evolved_state.state) == -1j * (
        manual_composite_H @ bell_state_dm.state
        - bell_state_dm.state @ manual_composite_H
    )


def test_evolve_ket(composite_hamiltonian, manual_composite_H):
    evolved_state = composite_hamiltonian.onState(bell_state_ket)
    assert (
        pytest.approx(evolved_state.state)
        == -1j * manual_composite_H @ bell_state_ket.state
    )


@pytest.fixture
def time_dependent_hamiltonian_generator():
    return HamiltonianGenerator(sigmaX + np.sin * TOperator.from_static(sigmaY))


def test_time_dependent_dynamics(time_dependent_hamiltonian_generator):
    H = sigmaX + np.sin(3) * sigmaY
    assert (
        pytest.approx(
            time_dependent_hamiltonian_generator.onState(spinUpDensityMatrix, 3).state
        )
        == -1j * H.commutator(spinUpDensityMatrix).state
    )
