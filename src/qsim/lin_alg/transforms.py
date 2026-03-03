import numpy as np

from .operator import Operator
from .vector import Vector


def vectorise(rho: Operator) -> Vector:
    return Vector(rho.matrix.reshape(-1, order="F"))


@staticmethod
def unvectorise(s: Vector) -> Operator:
    d = int(np.sqrt(s.dim))
    return Operator(s.matrix.reshape((d, d), order="F"))
