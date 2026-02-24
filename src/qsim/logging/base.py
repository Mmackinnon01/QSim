from numbers import Real
from re import S
from typing import Dict, Self, Set

from qsim.state.base import QuantumState
from qsim.state.detector import Detector


class Logger:

    def __init__(
        self, detectors: list[Detector] | None = None, log_state=False
    ) -> None:
        self._detectors: list[Detector] = detectors if detectors else []
        self._log_state: bool = log_state
        self.clear()

    def log(self, state: QuantumState, t: Real) -> Self:
        if t in self.times:
            raise ValueError(f"Cannot log t={t} as it has already been logged")
        self.times.add(t)

        if self._log_state:
            self.state_log[t] = state

        for detector in self._detectors:
            self.observable_log[detector][t] = detector.detect(state)

        return self

    def clear(self):
        self.times: Set[Real] = set()
        self.state_log: Dict[Real, QuantumState] = {}
        self.observable_log: Dict[Detector, Dict[Real, Real]] = {
            d: {} for d in self._detectors
        }
