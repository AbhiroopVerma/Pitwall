"""Module 4 — own-car pace model (asymmetric info).

Uses the same model as the opponent, so neither car gets a built-in advantage.
The extra information: when a private wear signal is available, the degradation
curve is fitted to that signal instead of to noisy lap times.

In the demo race the signal is clearly labelled SYNTHETIC (see
``data.generate_race``). In a real team this slot takes private telemetry. On
historical races there is no signal, so the model falls back to lap times.
"""
from __future__ import annotations

import numpy as np

from .degradation import (DegradationEstimator, _fit_curve)
from .state import LapRecord


class OwnCarModel(DegradationEstimator):
    uses_private_signal: bool = False

    @property
    def std_(self) -> float:
        return self.resid_std_

    def _curve_target(self, recs: list[LapRecord], y: np.ndarray,
                      curve: np.ndarray, base: np.ndarray):
        wear = [r.private_wear for r in recs]
        if any(w is None for w in wear):
            self.uses_private_signal = False
            return _fit_curve(curve, base, y, self.ridge)
        self.uses_private_signal = True
        wear = np.asarray(wear, float)
        coef, _, _ = _fit_curve(curve, np.zeros((len(y), 0)), wear, self.ridge)
        # With the curve known from the signal, only stint baselines remain.
        rest = y - curve @ coef
        intercepts = np.linalg.lstsq(base, rest, rcond=None)[0]
        _, _, cov = _fit_curve(curve, base, y, self.ridge)
        return coef, intercepts, cov

    def predict_stay(self, tyre_age_now: int, from_lap: int, steps: int) -> list[float]:
        """Forecast if we stay out (age keeps growing)."""
        return [self.pace(tyre_age_now + k, from_lap + k, self.compound_)
                for k in range(1, steps + 1)]

    def predict_fresh(self, from_lap: int, steps: int) -> list[float]:
        """Forecast after boxing at the end of ``from_lap``, excluding the stop cost."""
        return [self.pace(k, from_lap + k) for k in range(1, steps + 1)]
