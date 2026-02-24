from multiprocessing import Value

import numpy as np
import pytest

from qsim.logging import Logger
from qsim.operator import sigmaX, sigmaZ
from qsim.state.density_matrix import DensityMatrix
from qsim.state.detector import Detector
from qsim.state.wave_vector import Ket

spin_down_dm = DensityMatrix(np.array([[1, 0], [0, 0]]))
spin_up_dm = DensityMatrix(np.array([[0, 0], [0, 1]]))
spin_down_ket = Ket(np.array([1, 0]))

det1 = Detector(sigmaX)
det2 = Detector(sigmaZ)


def test_log_state_only():
    assert Logger(log_state=True).log(spin_down_dm, t=3).state_log[3] == spin_down_dm
    assert Logger(log_state=True).log(spin_down_ket, t=3).state_log[3] == spin_down_ket


def test_log_observable_only():
    assert (
        Logger(detectors=[det1, det2]).log(spin_down_dm, t=3).observable_log[det1][3]
        == 0
    )
    assert (
        Logger(detectors=[det1, det2]).log(spin_down_ket, t=3).observable_log[det2][3]
        == 1
    )


def test_log_multiple():
    l = Logger(detectors=[det2], log_state=True)
    l.log(spin_down_dm, 1)
    l.log(spin_up_dm, 2)
    assert len(l.state_log) == 2
    assert len(l.observable_log) == 1
    assert len(l.observable_log[det2]) == 2
    assert l.observable_log[det2][2] == -1


def test_log_same_time_twice_invalid():
    l = Logger(log_state=True)
    l.log(spin_down_dm, 1)
    with pytest.raises(ValueError):
        l.log(spin_down_dm, 1)


def test_clear():
    l = Logger(detectors=[det2], log_state=True)
    l.log(spin_down_dm, 1)
    l.log(spin_up_dm, 2)
    l.clear()
    assert len(l.state_log) == 0
