import numpy as np
import pytest

from qsim.lin_alg import Operator, sigmaPlus, sigmaX, sigmaZ
from qsim.state import ObservableDetector, POVMDetector
from qsim.state.density_matrix import DensityMatrix
from qsim.state.wave_vector import Ket

bell_state_dm = DensityMatrix(
    np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) * 0.5
)
bell_state_ket = Ket(np.array([1, 0, 0, 1]) / 2**0.5)


@pytest.fixture
def povm_detector():
    return POVMDetector(
        povm=[
            Operator(np.array([[1, 0], [0, 0]])),
            Operator(np.array([[0, 0], [0, 1]])),
        ],
        outcomes=[1, -1],
    )


@pytest.fixture
def dual_povm_detector():
    return POVMDetector(
        povm=[
            Operator(np.array([[1, 0], [0, 0]])),
            Operator(np.array([[0, 0], [0, 1]])),
        ],
        outcomes=[1, -1],
        dims=(2, 2),
        target_sites=(1,),
    )


def test_non_observable_invalid():
    with pytest.raises(ValueError):
        ObservableDetector(sigmaPlus)


def test_measure_dm():
    d = ObservableDetector(observable=sigmaX.tensor(sigmaX))
    assert d.detect(bell_state_dm) == 1


def test_measure_dm_subsystem():
    d = ObservableDetector(observable=sigmaX, dims=(2, 2), target_sites=(1,))
    assert d.detect(bell_state_dm) == np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )


def test_measure_ket():
    d = ObservableDetector(observable=sigmaX.tensor(sigmaX))
    assert pytest.approx(d.detect(bell_state_ket)) == 1


def test_measure_ket_subsystem():
    d = ObservableDetector(observable=sigmaX, dims=(2, 2), target_sites=(1,))
    assert pytest.approx(d.detect(bell_state_ket)) == np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )


def test_measure_bra():
    d = ObservableDetector(observable=sigmaX.tensor(sigmaX))
    assert pytest.approx(d.detect(bell_state_ket.hConj())) == 1


def test_measure_bra_subsystem():
    d = ObservableDetector(observable=sigmaX, dims=(2, 2), target_sites=(1,))
    assert pytest.approx(d.detect(bell_state_ket.hConj())) == np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )


def test_finite_statistics():
    d = ObservableDetector(observable=sigmaX, dims=(2, 2), target_sites=(1,))
    assert pytest.approx(
        d.detect(bell_state_ket.hConj(), shots=10e10), abs=10e-6
    ) == np.trace(bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix))
    assert pytest.approx(
        d.detect(bell_state_ket @ bell_state_ket.hConj(), shots=10e10), abs=10e-6
    ) == np.trace(bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix))
    assert pytest.approx(d.detect(bell_state_ket.hConj(), shots=10e5)) != np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )
    assert pytest.approx(
        d.detect(bell_state_ket @ bell_state_ket.hConj(), shots=10e5)
    ) != np.trace(bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix))


def test_invalid_povm():
    with pytest.raises(ValueError):
        POVMDetector(
            povm=[
                Operator(np.array([[1.5, 0], [0, 0]])),
                Operator(np.array([[0, 0], [0, 1]])),
            ],
            outcomes=[1, -1],
        )


def test_measure_dm_povm(povm_detector):
    assert povm_detector.detect(DensityMatrix(np.array([[0.5, 0], [0, 0.5]]))) == 0


def test_measure_dm_povm_subsystem(dual_povm_detector):
    assert dual_povm_detector.detect(bell_state_dm) == 0


def test_measure_ket_povm(povm_detector):
    assert pytest.approx(povm_detector.detect(Ket(np.array([2**0.5, 2**0.5])))) == 0


def test_measure_ket_povm_subsystem(dual_povm_detector):
    assert pytest.approx(dual_povm_detector.detect(bell_state_ket)) == 0


def test_measure_bra_povm(povm_detector):
    assert (
        pytest.approx(povm_detector.detect(Ket(np.array([2**0.5, 2**0.5])).hConj()))
        == 0
    )


def test_measure_bra_povm_subsystem(dual_povm_detector):
    assert pytest.approx(dual_povm_detector.detect(bell_state_ket.hConj())) == 0


def test_finite_statistics_povm(povm_detector):
    k = Ket(np.array([1 / 2**0.5, 1 / 2**0.5]))
    assert pytest.approx(povm_detector.detect(k, shots=10e10), abs=10e-6) == 0
    assert (
        pytest.approx(povm_detector.detect(k @ k.hConj(), shots=10e10), abs=10e-6) == 0
    )
    assert pytest.approx(povm_detector.detect(k.hConj(), shots=10e5)) != np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )
    assert pytest.approx(povm_detector.detect(k @ k.hConj(), shots=10e5)) != 0
