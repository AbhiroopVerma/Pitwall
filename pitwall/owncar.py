"""Module 4 — own-car pace model (asymmetric info).

Same ridge engine as the opponent model BUT allowed one extra, clearly
labelled SYNTHETIC private signal: true tyre wear (tyre_age + small noise).
In a real team this slot takes real private telemetry; here it is synthetic
and must never be presented as measured data.
"""
from __future__ import annotations

import numpy as np

from .degradation import ridge_fit
from .state import LapRecord


class OwnCarModel:
    def __init__(self, ridge: float = 1.0):
        self.ridge = ridge
        self.coef_: np.ndarray | None = None
        self.std_: float = 0.25

    def fit(self, history: list[LapRecord], laps_total: int) -> "OwnCarModel":
        X, y = [], []
        for r in history:
            if r.pitted or r.safety_car:
                continue
            fuel = (laps_total - r.lap) / laps_total
            wear = r.tyre_age + np.sin(r.lap) * 0.05  # SYNTHETIC private wear proxy
            X.append([1.0, r.tyre_age, r.tyre_age ** 2, fuel, r.track_temp, wear])
            y.append(r.lap_time)
        X, y = np.asarray(X, float), np.asarray(y, float)
        if len(y) < 7:
            raise ValueError("not enough own-car laps")
        self.coef_ = ridge_fit(X, y, self.ridge)
        self.std_ = float(np.std(y - X @ self.coef_, ddof=1)) or 0.25
        return self

    def predict_stay(self, tyre_age_now: int, from_lap: int, laps_total: int,
                     track_temp: float, steps: int) -> list[float]:
        """Forecast if we stay out (age keeps growing)."""
        b0, b1, b2, b3, b4, b5 = self.coef_
        out = []
        for k in range(1, steps + 1):
            age = tyre_age_now + k
            fuel = (laps_total - (from_lap + k)) / laps_total
            out.append(float(b0 + b1 * age + b2 * age ** 2 + b3 * fuel + b4 * track_temp + b5 * age))
        return out

    def predict_fresh(self, from_lap: int, laps_total: int, track_temp: float,
                      steps: int, new_compound_bonus: float = -0.4) -> list[float]:
        """Forecast after pitting now (fresh tyre, age restarts at 1)."""
        b0, b1, b2, b3, b4, b5 = self.coef_
        out = []
        for k in range(1, steps + 1):
            fuel = (laps_total - (from_lap + k)) / laps_total
            out.append(float(b0 + b1 * k + b2 * k ** 2 + b3 * fuel
                             + b4 * track_temp + b5 * k + new_compound_bonus))
        return out
