from __future__ import annotations

import numpy as np

from .operator import Operator


class Observable(Operator):

    def __init__(self, matrix: np.ndarray | Operator) -> None:
        if isinstance(matrix, np.ndarray):
            matrix = Operator(matrix)

        if matrix.isSelfAdjoint():
            self.matrix = matrix.matrix
        else:
            raise ValueError("Cannot assign non-selfadjoint matrix to observable")
