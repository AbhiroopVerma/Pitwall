"""Module 2 — opponent tyre-degradation & pace estimation.

Model: L = b0 + b1*age + b2*age^2 + b3*fuel + b4*temp + eps   (ridge, numpy only)
Tyre penalty D(age) = b1*age + b2*age^2. Forecasts project age forward;
fuel is projected from laps remaining. Pit laps and safety-car laps are
excluded from the fit (they are not clean pace observations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import LapRecord


@dataclass
class DegradationEstimate:
    penalty: float        # D(age_now), seconds vs fresh
    slope: float          # dD/d_age at current age (s/lap)
    forecast: list[float]  # expected clean lap times for next k laps (stay-out)
    std: float            # residual std (uncertainty)
    max_extra_laps: int   # laps until penalty exceeds `limit` (default 2.0 s)


def _features(recs: list[LapRecord], laps_total: int):
    X, y = [], []
    for r in recs:
        if r.pitted or r.safety_car:
            continue
        fuel = (laps_total - r.lap) / laps_total
        X.append([1.0, r.tyre_age, r.tyre_age ** 2, fuel, r.track_temp])
        y.append(r.lap_time)
    return np.asarray(X, float), np.asarray(y, float)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    n_feat = X.shape[1]
    A = X.T @ X + lam * np.eye(n_feat)
    A[0, 0] -= lam  # don't penalise intercept
    return np.linalg.solve(A, X.T @ y)


class DegradationEstimator:
    def __init__(self, ridge: float = 1.0, horizon: int = 5):
        self.ridge = ridge
        self.horizon = horizon
        self.coef_: np.ndarray | None = None

    def fit(self, history: list[LapRecord], laps_total: int) -> "DegradationEstimator":
        X, y = _features(history, laps_total)
        if len(y) < 6:
            raise ValueError("not enough clean laps to fit degradation model")
        self.coef_ = ridge_fit(X, y, self.ridge)
        self._resid_std = float(np.std(y - X @ self.coef_, ddof=1)) or 0.25
        return self

    def estimate(self, history: list[LapRecord], laps_total: int,
                 current_lap: int, penalty_limit: float = 2.0) -> DegradationEstimate:
        assert self.coef_ is not None, "call fit() first"
        b0, b1, b2, b3, b4 = self.coef_
        cur = history[-1]
        age = cur.tyre_age
        penalty = b1 * age + b2 * age ** 2
        slope = b1 + 2 * b2 * age
        forecast = []
        for k in range(1, self.horizon + 1):
            fuel = (laps_total - (current_lap + k)) / laps_total
            forecast.append(float(b0 + b1 * (age + k) + b2 * (age + k) ** 2
                                  + b3 * fuel + b4 * cur.track_temp))
        extra = 0
        while b1 * (age + extra + 1) + b2 * (age + extra + 1) ** 2 < penalty_limit and extra < 15:
            extra += 1
        return DegradationEstimate(penalty, slope, forecast, self._resid_std, extra)
