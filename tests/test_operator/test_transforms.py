import numpy as np
import pytest

from qsim.lin_alg import Operator, Vector
from qsim.lin_alg.transforms import unvectorise, vectorise


def test_vectorise_and_unvectorise():
    s = Operator(np.array([[0.1, 0], [0, 0.9]]))
    assert pytest.approx(vectorise(s).matrix) == np.array([0.1, 0, 0, 0.9]).reshape(
        -1, 1
    )
    assert pytest.approx(unvectorise(vectorise(s)).matrix) == s.matrix
