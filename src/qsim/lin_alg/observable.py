from __future__ import annotations

import numpy as np

from .operator import Operator


class Observable(Operator):

    def __init__(self, matrix: np.ndarray | Operator) -> None:
        self.matrix = matrix.astype(np.complex128)
        if not self.isSelfAdjoint():
            raise ValueError("Cannot assign non-selfadjoint matrix to observable")
        super().__init__(matrix)
