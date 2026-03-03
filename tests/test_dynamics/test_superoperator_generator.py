import numpy as np
import pytest

from qsim.dynamics import GKSLGenerator, LiouvillianGenerator
from qsim.lin_alg import I, Operator, Vector, sigmaMinus, sigmaX
from qsim.lin_alg.transforms import vectorise


def test_constructs_from_gksl():
    gksl = GKSLGenerator(H=sigmaX, jumps=[sigmaMinus])
    l = LiouvillianGenerator.fromGKSL(gksl)
    assert (
        pytest.approx(l.L.matrix)
        == (
            -1j * (I(2).tensor(sigmaX) - sigmaX.tensor(I(2)))
            + sigmaMinus.tensor(sigmaMinus)
            - 0.5
            * (
                I(2).tensor(sigmaMinus.hConj() @ sigmaMinus)
                + (sigmaMinus.hConj() @ sigmaMinus).T.tensor(I(2))
            )
        ).matrix
    )


def test_spectral_decomposition():
    gksl = GKSLGenerator(H=I(2), jumps=[sigmaMinus])
    l = LiouvillianGenerator.fromGKSL(gksl)
    eigs, lv, rv = l.spectralDecomposition()
    assert 0 in eigs
    assert -0.5 in eigs
    assert -1 in eigs
    assert Vector(np.array([1, 0, 0, 0])) in rv
