import numpy as np
import pytest

from qsim.lin_alg import sigmaPlus, sigmaX, sigmaZ
from qsim.state import Detector
from qsim.state.density_matrix import DensityMatrix
from qsim.state.wave_vector import Ket

bell_state_dm = DensityMatrix(
    np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) * 0.5
)
bell_state_ket = Ket(np.array([1, 0, 0, 1]) / 2**0.5)


def test_non_observable_invalid():
    with pytest.raises(ValueError):
        Detector(sigmaPlus)


def test_measure_dm():
    d = Detector(observable=sigmaX.tensor(sigmaX))
    assert d.detect(bell_state_dm) == 1


def test_measure_dm_subsystem():
    d = Detector(observable=sigmaX, dims=(2, 2), target_sites=(1,))
    assert d.detect(bell_state_dm) == np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )


def test_measure_ket():
    d = Detector(observable=sigmaX.tensor(sigmaX))
    assert pytest.approx(d.detect(bell_state_ket)) == 1


def test_measure_ket_subsystem():
    d = Detector(observable=sigmaX, dims=(2, 2), target_sites=(1,))
    assert pytest.approx(d.detect(bell_state_ket)) == np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )


def test_measure_bra():
    d = Detector(observable=sigmaX.tensor(sigmaX))
    assert pytest.approx(d.detect(bell_state_ket.hConj())) == 1


def test_measure_bra_subsystem():
    d = Detector(observable=sigmaX, dims=(2, 2), target_sites=(1,))
    assert pytest.approx(d.detect(bell_state_ket.hConj())) == np.trace(
        bell_state_dm.matrix @ np.kron(np.eye(2), sigmaX.matrix)
    )
